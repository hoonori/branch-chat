import streamlit as st
import uuid

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

# --- Functions ---
def render_chat(branch):
    st.subheader("💬 Current Conversation")
    for node_id in branch:
        node = st.session_state.nodes[node_id]
        st.markdown(f"**You:** {node.user_input}")
        st.markdown(f"> {node.response}")

        if st.button(f"🔀 Branch from here", key=f"branch_{node.id}"):
            new_branch = branch[:branch.index(node_id)+1]
            st.session_state.current_branch = new_branch
            st.rerun()

def fake_llm_response(user_input):
    return f"This is a fake response to: '{user_input}'"

def add_message(user_input):
    response = fake_llm_response(user_input)
    parent_id = st.session_state.current_branch[-1]
    new_node = ChatNode(user_input, response, parent_id)
    st.session_state.nodes[new_node.id] = new_node
    st.session_state.nodes[parent_id].children.append(new_node.id)
    st.session_state.current_branch.append(new_node.id)

def render_tree_ui(current_id, depth=0):
    node = st.session_state.nodes[current_id]
    is_current = current_id in st.session_state.current_branch
    box_color = "#e0f7fa" if is_current else "#f0f0f0"
    indent = depth * 30

    st.markdown(
        f"""
        <div style='margin-left:{indent}px; padding:10px; border-radius: 10px; background-color:{box_color}; border: 1px solid #ccc;'>
            <b>User:</b> {node.user_input}<br>
            <b>Bot:</b> {node.response}<br>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Streamlit 버튼으로 대체 및 디버깅 로그 추가
    if st.button(f"Jump & Branch Here (debug)", key=f"jump_{node.id}"):
        st.write(f"[DEBUG] Jump & Branch Here clicked for node_id: {node.id}")
        # 브랜치 경로 계산
        branch = []
        cur = node.id
        while cur:
            branch.insert(0, cur)
            cur = st.session_state.nodes[cur].parent_id
        st.session_state.current_branch = branch
        st.write(f"[DEBUG] New branch: {branch}")
        st.write(f"[DEBUG] Session state keys: {list(st.session_state.keys())}")
        st.write(f"[DEBUG] Nodes: {list(st.session_state.nodes.keys())}")
        st.write(f"[DEBUG] Current branch: {st.session_state.current_branch}")
        st.rerun()

    for child_id in node.children:
        render_tree_ui(child_id, depth + 1)

# --- Layout ---
col1, col2 = st.columns([2, 1])

with col1:
    render_chat(st.session_state.current_branch)
    user_input = st.text_input("Your message:", key="user_input")
    if st.button("Send") and user_input:
        add_message(user_input)
        st.rerun()

with col2:
    st.subheader("🧠 Global Context (simulated)")
    for i, ctx in enumerate(global_context):
        st.markdown(f"**{i+1}.** {ctx}")

    if st.button("➕ Add current input to Global Context"):
        last_node = st.session_state.nodes[st.session_state.current_branch[-1]]
        global_context.append(last_node.user_input + " / " + last_node.response)
        st.rerun()

    st.divider()
    st.subheader("🌳 Conversation Tree View")
    render_tree_ui(st.session_state.root_id)

st.caption("🔧 This is a prototype. LLM integration and mem0 RAG planned later.")
