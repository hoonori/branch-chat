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

def render_branch_selector():
    st.subheader("🌳 Conversation Tree")
    branch_paths = []

    def dfs(path):
        current = path[-1]
        node = st.session_state.nodes[current]
        if not node.children:
            branch_paths.append(path)
        else:
            for child_id in node.children:
                dfs(path + [child_id])

    dfs([st.session_state.root_id])

    for idx, path in enumerate(branch_paths):
        label = " → ".join([st.session_state.nodes[nid].user_input[:15] for nid in path])
        if st.button(f"🔁 Switch to branch #{idx+1}: {label}", key=f"switch_{idx}"):
            st.session_state.current_branch = path
            st.rerun()

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
    render_branch_selector()

st.caption("🔧 This is a prototype. LLM integration and mem0 RAG planned later.")
