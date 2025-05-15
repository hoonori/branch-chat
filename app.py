import streamlit as st
import uuid
import streamlit.components.v1 as components
from openai_chat import chat_with_gpt4o
from global_context import GlobalRAGContext
import pyperclip

st.set_page_config(page_title="Branchable Chat", layout="wide")

# --- Simulated RAG Context Memory ---
global_context = []  # Eventually replaced by mem0

# --- Chat Node Structure ---
class ChatNode:
    def __init__(self, user_input, response, parent_id=None):
        self.id = str(uuid.uuid4())
        self.parent_id = parent_id
        self.user_input = user_input
        self.response = response
        self.children = []

# --- Session State Initialization ---
if "nodes" not in st.session_state:
    root = ChatNode("Hello! What can I help you with today?", "Hi! Ask me anything.")
    st.session_state.nodes = {root.id: root}
    st.session_state.root_id = root.id
    st.session_state.current_branch = [root.id]
# RAG context 세션에 저장
if "rag_context" not in st.session_state:
    st.session_state.rag_context = GlobalRAGContext()
# Jump history 세션에 저장
if "jump_history" not in st.session_state:
    st.session_state.jump_history = []

# --- Jump to Branch Helper ---
def jump_to_branch(new_branch):
    # 점프 전 현재 브랜치를 history에 저장 (pinned=False)
    st.session_state.jump_history.append({"branch": list(st.session_state.current_branch), "pinned": False})
    st.session_state.current_branch = new_branch
    st.rerun()

# --- Functions ---
def render_chat(branch):
    st.subheader("💬 Current Conversation")
    for node_id in branch:
        node = st.session_state.nodes[node_id]
        st.markdown(f"**You:** {node.user_input}")
        st.markdown(f"> {node.response}")
        # 참고 노드 정보 표시
        if hasattr(node, "rag_refs") and node.rag_refs:
            st.markdown("<div style='margin-left:20px; color:#888;'>참고한 노드:</div>", unsafe_allow_html=True)
            for i, ref in enumerate(node.rag_refs):
                with st.container():
                    st.markdown(f"<div style='margin-left:30px; border:1px solid #eee; border-radius:8px; padding:6px 10px; background:#f8f8fa; font-size:13px;'>"
                        f"<b>질문:</b> {ref['question']}<br>"
                        f"<b>답변:</b> {ref['answer']}<br>"
                        f"<b>노드ID:</b>", unsafe_allow_html=True)
                    st.code(ref['node_id'], language=None)
                    st.caption("위의 Copy 버튼을 눌러 클립보드에 복사하세요.")
                    st.markdown(f"<b>시간:</b> {ref['timestamp']}", unsafe_allow_html=True)
        if st.button(f"🔀 Branch from here", key=f"branch_{node.id}"):
            new_branch = branch[:branch.index(node_id)+1]
            jump_to_branch(new_branch)

# 실제 LLM 호출 함수 (RAG 참고)
def llm_response(user_input):
    branch = st.session_state.current_branch
    nodes = st.session_state.nodes
    rag = st.session_state.rag_context
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    for node_id in branch[1:]:
        node = nodes[node_id]
        messages.append({"role": "user", "content": node.user_input})
        messages.append({"role": "assistant", "content": node.response})
    messages.append({"role": "user", "content": user_input})
    # RAG: 유사 노드 top-3
    rag_refs = rag.search(user_input, top_k=3)
    # 참고 노드 정보를 프롬프트에 추가
    if rag_refs:
        context_str = "\n\n".join([f"Q: {doc.question}\nA: {doc.answer}" for doc in rag_refs])
        messages.append({"role": "system", "content": f"참고할 만한 이전 대화들:\n{context_str}"})
    response = chat_with_gpt4o(messages)
    return response, rag_refs

def add_message(user_input):
    response, rag_refs = llm_response(user_input)
    parent_id = st.session_state.current_branch[-1]
    new_node = ChatNode(user_input, response, parent_id)
    st.session_state.nodes[new_node.id] = new_node
    st.session_state.nodes[parent_id].children.append(new_node.id)
    st.session_state.current_branch.append(new_node.id)
    # RAG 문서로 저장
    rag = st.session_state.rag_context
    rag.add_document(user_input, response, new_node.id)
    # 참고 노드 정보 저장
    new_node.rag_refs = [
        {"question": doc.question, "answer": doc.answer, "node_id": doc.node_id, "timestamp": doc.timestamp}
        for doc in rag_refs
    ]

def render_tree_svg_ui():
    import math
    nodes = st.session_state.nodes
    root_id = st.session_state.root_id
    current_branch = st.session_state.current_branch
    levels = {}
    positions = {}
    def traverse(node_id, depth=0, x=0):
        if depth not in levels:
            levels[depth] = []
        levels[depth].append(node_id)
        y = depth * 120 + 60
        positions[node_id] = [x, y]
        child_x = x
        for child_id in nodes[node_id].children:
            child_x = traverse(child_id, depth+1, child_x)
            child_x += 180
        if nodes[node_id].children:
            first = positions[nodes[node_id].children[0]][0]
            last = positions[nodes[node_id].children[-1]][0]
            positions[node_id][0] = (first + last) // 2
        return positions[node_id][0]
    traverse(root_id)
    all_x = [pos[0] for pos in positions.values()]
    all_y = [pos[1] for pos in positions.values()]
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    width = max(600, max_x - min_x + 200)
    height = max(400, max_y - min_y + 100)
    svg = f'<svg id="tree-svg" width="{width}" height="{height}" style="background:#fafcff;">'
    for node_id, node in nodes.items():
        x1, y1 = positions[node_id]
        for child_id in node.children:
            x2, y2 = positions[child_id]
            svg += f'<line x1="{x1+80}" y1="{y1+40}" x2="{x2+80}" y2="{y2}" stroke="#bbb" stroke-width="2" />'
    for node_id, node in nodes.items():
        x, y = positions[node_id]
        is_current = node_id in current_branch
        color = "#e0f7fa" if is_current else "#fff"
        border = "#0097a7" if is_current else "#bbb"
        svg += f'<g>'
        svg += f'<rect x="{x+40}" y="{y}" width="80" height="60" rx="12" fill="{color}" stroke="{border}" stroke-width="3" style="cursor:pointer;" onclick="sendNodeId(\'{node_id}\')" />'
        svg += f'<text x="{x+80}" y="{y+25}" text-anchor="middle" font-size="13" fill="#333" pointer-events="none">{node.user_input[:10]}</text>'
        svg += f'<text x="{x+80}" y="{y+45}" text-anchor="middle" font-size="11" fill="#888" pointer-events="none">{node.response[:10]}</text>'
        svg += f'</g>'
    svg += '</svg>'
    html = f'''
    <script>
    function sendNodeId(node_id) {{
        navigator.clipboard.writeText(node_id);
        // alert("노드 id가 복사되었습니다! 아래 입력창에 붙여넣고 Jump를 누르세요.");
        window.parent.postMessage({{isStreamlitMessage: true, type: "SVG_JUMP", node_id: node_id}}, "*");
    }}
    </script>
    {svg}
    '''
    components.html(html, height=height+20, scrolling=False)
    # 노드 id 입력창 + Jump 버튼
    st.markdown("#### 노드 id로 브랜치 점프")
    node_id_input = st.text_input("노드 id를 입력하세요(노드 클릭 후 붙여넣기)", key="node_id_input")
    if st.button("Jump to Node"):
        # 점프 전 현재 브랜치를 history에 저장
        branch = []
        cur = node_id_input
        nodes = st.session_state.nodes
        if node_id_input in nodes:
            while cur:
                branch.insert(0, cur)
                cur = nodes[cur].parent_id
            jump_to_branch(branch)
        else:
            st.warning("존재하지 않는 노드 id입니다.")

# --- Jump History UI ---
def render_jump_history():
    st.subheader("⬅️ 이전 위치로 돌아가기 (Jump History)")
    history = st.session_state.jump_history
    if history:
        # 고정된 엔트리 먼저, 그 다음 일반 엔트리
        pinned = [h for h in history if h["pinned"]]
        normal = [h for h in history if not h["pinned"]]
        show_list = pinned + normal
        for idx, entry in enumerate(show_list):
            branch = entry["branch"]
            last_node_id = branch[-1]
            node = st.session_state.nodes[last_node_id]
            label = f"{node.user_input[:20]} | {node.response[:20]}"
            cols = st.columns([6,1,1])
            with cols[0]:
                if st.button(f"이 위치로 돌아가기: {label}", key=f"jump_back_{idx}"):
                    st.session_state.current_branch = entry["branch"]
                    # 점프 후, 고정이 아니면 pop
                    if not entry["pinned"]:
                        st.session_state.jump_history.pop(history.index(entry))
                    st.rerun()
            with cols[1]:
                pin_label = "고정해제" if entry["pinned"] else "고정"
                if st.button(pin_label, key=f"pin_{idx}"):
                    entry["pinned"] = not entry["pinned"]
                    st.rerun()
            with cols[2]:
                if st.button("삭제", key=f"del_{idx}"):
                    st.session_state.jump_history.pop(history.index(entry))
                    st.rerun()
    else:
        st.caption("점프 이력이 없습니다.")

# --- Layout ---
col1, col2 = st.columns([2, 1])

with col1:
    render_jump_history()
    render_chat(st.session_state.current_branch)
    user_input = st.text_input("Your message:", key="user_input")
    if st.button("Send") and user_input:
        add_message(user_input)
        st.rerun()

with col2:
    st.subheader("🌳 Conversation Tree View (SVG)")
    render_tree_svg_ui()

st.caption("🔧 This is a prototype. LLM integration and mem0 RAG planned later.")
