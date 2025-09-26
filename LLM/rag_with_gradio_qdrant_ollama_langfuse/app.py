import gradio as gr
import time
from utils import *
import uuid
from langfuse.decorators import langfuse_context, observe
from langfuse import Langfuse
from collections import defaultdict
import re

# ==================== GLOBALS ====================

# Session and trace tracking
session_id = None                 # active session id for new chats
trace_ids = {}                    # map chat message index -> trace_id for like/retry
current_selected_session_id = None  # if user opened an old conversation, use its session_id
last_assistant_trace_id = None
citation_trace_map = {}

langfuse_client = Langfuse()

# ==================== USER IDENTIFICATION ====================

def get_user_info(request: gr.Request = None):
    """
    Retrieve user info from request headers (adapted for Gradio).
    Looks for 'x-appservice-muid' header. Falls back to UnregisteredLocalUser.
    """
    headers_ctx = request.headers if (request and hasattr(request, "headers")) else {}
    identifier = headers_ctx.get("x-appservice-muid")    
    if identifier is None:
        return {
            "identifier": "UnregisteredLocalUser",
            "role": "user",
            "provider": "header",
        }
    else:
        return {
            "identifier": identifier,
            "role": "user",
            "provider": "header",
        }

# ==================== SESSION MGMT ====================

def set_new_session_id():
    global session_id, trace_ids, last_assistant_trace_id, citation_trace_map
    session_id = str(uuid.uuid4())
    trace_ids = {}
    last_assistant_trace_id = None
    citation_trace_map = {}

# Initialize first session
set_new_session_id()

def get_active_session_id():
    """Use selected session if any; otherwise use the current new-session id."""
    return current_selected_session_id or session_id

# ==================== HELPERS ====================

def get_message_data(msg):
    """Extract role and content from either gr.ChatMessage or dict."""
    if hasattr(msg, 'role') and hasattr(msg, 'content'):
        return msg.role, msg.content, getattr(msg, 'metadata', None)
    elif isinstance(msg, dict):
        return msg.get('role'), msg.get('content', ''), msg.get('metadata')
    else:
        return None, None, None

def _first_user_message(items):
    """
    Return the first user 'input' from the oldest trace in a session,
    then standardized the title length to be displayed in the sidebar.
    This is the way to work with gradio I found, other dummy spaces and css would not work. 
    """
    EN_SPACE = "\u2002"  # Unicode EN SPACE

    try:
        # Sort items by timestamp in ascending order to get the oldest trace first
        items_sorted = sorted(items, key=lambda x: x.get("timestamp", ""))
    except Exception:
        items_sorted = items

    # Pick the oldest trace and extract its user input
    if items_sorted:
        oldest_trace = items_sorted[0]
        if oldest_trace.get("input"):
            txt = re.sub(r"\s+", " ", str(oldest_trace["input"])).strip()
            if len(txt) > 24:
                return txt[:24] + "…"
            else:
                # Pad with EN SPACE to reach 25 characters
                return txt + (EN_SPACE * (25 - len(txt)))
    # If no user input found, return 25 EN SPACEs
    return EN_SPACE * 25

def build_session_options(user_identifier: str, limit: int = 50):
    try:
        traces_batch = langfuse_client.fetch_traces(tags=f"{user_identifier}").data
    except Exception as e:
        print(f"Langfuse fetch_traces failed: {e}")
        return [], {"label_to_sid": {}, "sessions": {}}

    grouped = defaultdict(list)
    for tr in traces_batch or []:
        sid = getattr(tr, "session_id", None) or "no-session"
        info = {
            "session_id": sid,
            "trace_id": getattr(tr, "id", ""),
            "timestamp": str(getattr(tr, "timestamp", "")),
            "input": getattr(tr, "input", ""),
            "output": getattr(tr, "output", ""),
        }
        grouped[sid].append(info)

    sessions_items = list(grouped.items())

    # Sort sessions by latest timestamp desc
    sessions_items.sort(key=lambda kv: max((x.get("timestamp") or "") for x in kv[1]) if kv[1] else "", reverse=True)

    labels = []
    label_to_sid = {}
    sessions = {}

    for sid, items in sessions_items:
        label = _first_user_message(items) or sid
        labels.append(label)
        label_to_sid[label] = sid
        sessions[sid] = items

    return labels, {"label_to_sid": label_to_sid, "sessions": sessions}

def conv_to_messages_from_traces(traces: list):
    """
    Convert Langfuse traces list (for a single session_id) into Chatbot messages
    AND build a mapping from assistant message index -> trace_id.
    """
    try:
        traces = sorted(traces, key=lambda x: x.get("timestamp", ""))
    except Exception:
        pass

    msgs = []
    idx_to_trace = {}

    for t in traces:
        user_input = t.get("input", "")
        assistant_output = t.get("output", "")
        trace_id = t.get("trace_id") or t.get("id")  # be tolerant to structure

        if user_input:
            msgs.append(gr.ChatMessage(role="user", content=user_input))

        if assistant_output:
            # The assistant message will be appended at this index
            assistant_idx = len(msgs)
            msgs.append(gr.ChatMessage(role="assistant", content=assistant_output))
            if trace_id:
                idx_to_trace[assistant_idx] = trace_id

    return msgs, idx_to_trace


def user(user_message, history: list):
    """
    Add the user's input to the conversation history.
    """
    history.append(gr.ChatMessage(role="user", content=user_message))
    time.sleep(0.25)
    yield "", history

def handle_empty_input(history):
    """
    Handle the case where the user input is empty.
    """
    empty_message_response = "何も入力されていません。質問を入力してください。"
    update_langfuse_context(output=empty_message_response)
    history.append(gr.ChatMessage(role="assistant", content=empty_message_response))
    return history

def handle_error_in_response(answer, history):
    """
    Handle error detection in response using specific error patterns.
    """
    error_patterns = ["🚨 エラーが発生しました。開発者に連絡してください。できるだけ早く対応いたします。"]
    
    if any(pattern in answer.lower() for pattern in error_patterns):
        error_response = "🚨 エラーが発生しました。開発者に連絡してください。できるだけ早く対応いたします。🚨 \nご連絡は henri.defretin@merckgroup.com までお願いします。"
        
        # Update Langfuse context with error
        update_langfuse_context(
            output=error_response, 
            metadata={"error": "Error detected in response", "original_answer": answer}
        )
        
        # Replace the last assistant message with the error message
        if history and history[-1].role == 'assistant':
            history[-1] = gr.ChatMessage(role="assistant", content=error_response)
        else:
            history.append(gr.ChatMessage(role="assistant", content=error_response))
        
        return True, history
    else:
        return False, history


def append_metadata(history, answer, metadata_chunks, total_time, first_token_time):
    """
    Format and append metadata to the conversation history.
    """
    if answer not in ['この文脈には関連情報がありません。', 'There is no information in this context.']:
        # Format metadata
        formatted_metadata = ""
        if metadata_chunks:
            for chunk in metadata_chunks:
                if "Document:" in chunk:
                    formatted_metadata += f"{chunk}\n"

        formatted_metadata += (
            f"最初のトークンまでの時間: {first_token_time:.2f} 秒\n"
            f"合計生成時間: {total_time:.2f} 秒"
        )

        # Append metadata to history
        history.append(gr.ChatMessage(
            role="assistant",
            content=formatted_metadata,
            metadata={"title": "🛠️ 答えを生成するために使用されたデータ"}
        ))
    return history

async def warn_if_slow_start(started_evt: asyncio.Event, finished_evt: asyncio.Event, timeout: float):
    try:
        await asyncio.sleep(timeout)
        # Only warn if we STILL haven't started streaming and we haven't finished
        if not started_evt.is_set() and not finished_evt.is_set():
            gr.Warning("👺 応答の準備に時間がかかっています。少々お待ちください… 👺")
    except asyncio.CancelledError:
        pass

def update_langfuse_context(input=None, output=None, metadata=None):
    """
    Handle all Langfuse context updates for input, output, and metadata.
    Note: do not pass session_id here; the observation is already bound to the current trace.
    """
    try:
        update_data = {}
        if input is not None:
            update_data["input"] = input
        if output is not None:
            update_data["output"] = output
        if metadata is not None:
            update_data["metadata"] = metadata
        if update_data:
            langfuse_context.update_current_observation(**update_data)
    except Exception as e:
        print(f"Langfuse update_current_observation failed: {e}")

# ==================== SIDEBAR HANDLERS ====================

def refresh_conv_selector(limit: int = 50, request: gr.Request = None):
    """
    Refresh conversation selector (sessions grouped by session_id) from Langfuse.
    Only used on app load and after clearing chat.
    """
    user_info = get_user_info(request)
    user_identifier = user_info["identifier"]
    labels, mapping = build_session_options(user_identifier, limit)
    return gr.update(choices=labels, value=None), mapping

def on_select_conv(label: str, mapping: dict):
    """
    When a session is selected, load its traces and rebuild trace_ids so like/retry work.
    """
    global current_selected_session_id, session_id, trace_ids, last_assistant_trace_id

    if not label or not mapping:
        current_selected_session_id = None
        trace_ids = {}  # clear mapping
        return gr.update(value=[])

    sid = (mapping.get("label_to_sid") or {}).get(label)
    if not sid:
        current_selected_session_id = None
        trace_ids = {}  # clear mapping
        return gr.update(value=[])

    current_selected_session_id = sid
    session_id = sid

    session_traces = (mapping.get("sessions") or {}).get(sid, [])

    # Build messages + assistant index -> trace_id map
    msgs, idx_to_trace = conv_to_messages_from_traces(session_traces)

    # IMPORTANT: replace the global mapping so likes/retries target the right trace
    trace_ids = idx_to_trace

    # Track the latest assistant trace for fallback attribution of clicks
    if idx_to_trace:
        try:
            last_assistant_trace_id = idx_to_trace[max(idx_to_trace.keys())]
        except Exception:
            last_assistant_trace_id = None
    else:
        last_assistant_trace_id = None

    return gr.update(value=msgs)


# ==================== CITATION HANDLERS ====================

async def handle_citation_link_click(qdrant_id: str):
    citation_text, document, number = retrieve_chunk_content_by_id(qdrant_id)
    citation_content = citation_text.split("## Keywords")[0]    # Only show actual content of citation without metadata/keywords
    citation_content_escaped = escape_html_like_substrings(citation_content)

    # Determine which trace to attribute the click to
    trace_id_for_click = citation_trace_map.get(qdrant_id)
    if not trace_id_for_click:
        trace_id_for_click = last_assistant_trace_id
    if not trace_id_for_click and trace_ids:
        try:
            # Use the latest assistant message's trace as a last resort
            latest_idx = max(k for k in trace_ids.keys() if isinstance(k, int))
            trace_id_for_click = trace_ids.get(latest_idx)
        except Exception:
            trace_id_for_click = None

    # Record a score (+1) for this citation click in Langfuse
    # Use a unique score id so each click is counted
    if trace_id_for_click:
        try:
            score_id = f"cit-{trace_id_for_click}-{uuid.uuid4()}"
            # Reuse the same method style as the like button
            langfuse.score(
                id=score_id,
                name="citation-click",
                value=1,
                trace_id=trace_id_for_click
            )
        except Exception as e:
            print(f"Langfuse score for citation click failed: {e}")

    return f"# 参照: {document} - {number}\n\n{citation_content_escaped}"

def refresh_citation_display():
    """Reset/clear the citation panel."""
    return gr.update(value="## 参照\n\n参照を選択してください。")

# ==================== CHAT HANDLERS ====================

def clear_chat():
    """Clear current chat and reset to a brand-new session."""
    global current_selected_session_id
    current_selected_session_id = None
    set_new_session_id()
    return []

def user(user_message, history: list):
    """Append user's message to history."""
    history.append(gr.ChatMessage(role="user", content=user_message))
    yield "", history

@observe(name="Chatbot")
async def bot(prompt: str, history: list, request: gr.Request = None):
    """
    Main bot logic to process user input and generate assistant responses with real streaming.
    Assumes process_question is an async generator yielding text chunks, followed by:
      - a sentinel line "Metadata:"
      - then one or more metadata/citation lines
    """
    global trace_ids

    # Identify user and ensure trace has correct session/user/tags
    user_info = get_user_info(request)
    user_identifier = user_info["identifier"]
    active_sid = get_active_session_id()

    try:
        current_trace_id = langfuse_context.get_current_trace_id()
        if not current_trace_id:
            # Create new trace if none exists
            trace = langfuse_client.trace(
                session_id=active_sid,
                user_id=user_identifier,
                tags=[user_identifier, "Chatbot"]
            )
            current_trace_id = trace.id
        
        # Store trace ID before processing messages
        message_index = len(history)  # Index where assistant message will be
        trace_ids[message_index] = current_trace_id
        
        langfuse_client.trace(id=current_trace_id).update(
            session_id=active_sid,
            user_id=user_identifier,
            tags=[user_identifier, "Chatbot"],
            name="Chatbot"
        )
        
        # Track last assistant trace for citation clicks
        global last_assistant_trace_id
        last_assistant_trace_id = current_trace_id

    except Exception as e:
        print(f"Error updating current trace: {e}")

    # Get last user message (Gradio passes the message via history)
    last_user_message = ""
    for msg in reversed(history):
        role, content, _ = get_message_data(msg)
        if role == "user" and content:
            last_user_message = content
            break

    # Record input immediately (so early exits are still logged)
    update_langfuse_context(input=last_user_message)

    # Empty input handling
    if not last_user_message.strip():
        history = handle_empty_input(history)
        yield history
        return

    # Streaming state
    start_time = time.time()
    first_token_time = None
    answer = ""
    metadata_chunks = []
    assistant_message_added = False
    metadata_started = False

    started_evt = asyncio.Event()
    finished_evt = asyncio.Event()

    # Schedule the slow-start warning
    warning_task = asyncio.create_task(warn_if_slow_start(started_evt, finished_evt, timeout=15))

    # Stream chunks directly from the LLM pipeline (no fake slicing)
    async for chunk in process_question(last_user_message, return_complete=False):
        # Detect the metadata phase
        if chunk.strip() == "Metadata:":
            metadata_started = True
            continue

        if metadata_started:
            # Collect metadata/citation lines for the post-answer panel
            metadata_chunks.append(chunk.strip())

            # Map any qid=... found in metadata lines to this trace
            try:
                for m in re.finditer(r"[?&]qid=([^&\s]+)", chunk):
                    citation_trace_map[m.group(1)] = current_trace_id
            except Exception:
                pass

            continue

        # First token timing
        if first_token_time is None:
            first_token_time = time.time() - start_time
            if not started_evt.is_set():
                started_evt.set()
            if warning_task and not warning_task.done():
                warning_task.cancel()

        # Append streamed text directly
        answer += chunk
        try:
            for m in re.finditer(r"[?&]qid=([^&\s]+)", chunk):
                citation_trace_map[m.group(1)] = current_trace_id
        except Exception:
            pass
        if not assistant_message_added:
            history.append(gr.ChatMessage(role="assistant", content=""))
            assistant_message_added = True

        history[-1] = gr.ChatMessage(role="assistant", content=answer)
        yield history

    # If the LLM pipeline signaled an error, replace the assistant message
    error_detected, history = handle_error_in_response(answer, history)
    if error_detected:
        # Mark finished and cancel any pending warning
        if not finished_evt.is_set():
            finished_evt.set()
        if warning_task and not warning_task.done():
            warning_task.cancel()
        yield history
        return

    # Final timings
    end_time = time.time()
    total_time = end_time - start_time

    # Update trace/observation with output and timings
    update_langfuse_context(
        output=answer,
        metadata={
            "raw_metadata": metadata_chunks,
            "time_to_first_token": first_token_time,
            "total_generation_time": total_time
        }
    )

    # Append a separate metadata panel message (if applicable)
    history = append_metadata(history, answer, metadata_chunks, total_time, first_token_time)

    # Mark finished and cancel any pending warning
    if not finished_evt.is_set():
        finished_evt.set()
    if warning_task and not warning_task.done():
        warning_task.cancel()
    
    yield history

async def respond(prompt: str, history, request: gr.Request = None):
    """
    Wrapper function for bot responses.
    """
    async for message in bot(prompt, history, request):
        yield message

def handle_like(data: gr.LikeData):
    """
    Handle user feedback (like/dislike) for assistant responses.
    """
    trace_id = trace_ids.get(data.index)
    if trace_id:
        if data.liked:
            langfuse.score(id=f"like-{trace_id}", name="user-feedback", value=1, trace_id=trace_id)
        else:
            langfuse.score(id=f"like-{trace_id}", name="user-feedback", value=0, trace_id=trace_id)

async def handle_retry(retry_data: gr.RetryData, history, request: gr.Request = None):
    """Handle message retry in the active session (new or selected old)."""
    previous_prompt = ""
    if retry_data.index < len(history):
        role, content, metadata = get_message_data(history[retry_data.index])
        previous_prompt = content or ""
    
    # Mark the corresponding trace as retry (if available)
    trace_id = trace_ids.get(retry_data.index)
    if trace_id:
        try:
            langfuse_client.trace(id=trace_id).update(tags=["Chatbot", "retry"])
        except Exception as e:
            print(f"Error updating trace with retry tag: {e}")
    
    async for message in respond(previous_prompt, history, request):
        yield message

# ==================== GRADIO UI ====================

css = """
footer {visibility: hidden}
/* Make input container buttons smaller and left-aligned */
#input_container > .row { justify-content: flex-start !important; gap: 5px !important; }
#input_container button { flex: 0 0 auto !important; padding: 6px 12px !important; height: 32px !important; font-size: 12px !important; width: auto !important; min-width: 60px !important; max-width: 80px !important; }
.clear-button, button[aria-label="Clear"], button[title="Clear"] { display: none !important; }
button:has(svg[data-testid="trash"]), button[aria-label*="clear" i], button[title*="clear" i] { display: none !important; }
#chatbot { border: none !important; height: calc(100vh - 150px) !important; overflow-y: auto !important; padding-bottom: 10px !important; margin-bottom: 10px !important; flex-grow: 1 !important; }
#input_container { border: 1px solid #e0e0e0; border-radius: 10px; padding: 6px 15px 8px 15px !important; background: white; position: sticky !important; bottom: 24px !important; z-index: 100 !important; display: flex !important; flex-direction: column !important; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); max-width: 60% !important; margin: 10px auto !important; flex-shrink: 0 !important; }
#textbox { border-color: #FFFFFF; width: 100% !important; margin-bottom: 0 !important; }
.gradio-column { gap: 10px !important; height: 100% !important; display: flex !important; flex-direction: column !important; }
.gradio-row { gap: 10px !important; }
.gradio-container { margin-left: 240px !important; width: calc(100% - 240px) !important; height: 100vh !important; display: flex !important; flex-direction: column !important; }
.gradio-container .gradio-column { display: flex !important; flex-direction: column !important; height: 100vh !important; }
.gradio-container > .gradio-row > .gradio-column:nth-child(2) { height: 100vh !important; display: flex !important; flex-direction: column !important; }
.form { border: none !important; box-shadow: none !important; background: #FAFAFA }
#gradio-radio * { border: none !important; box-shadow: none !important; background: #FAFAFA }
.svelte-1bx8sav:hover { background: #E8E8E8 !important; }
.svelte-1bx8sav > input[type="radio"] { display:none }
#gradio-radio { border: none !important; box-shadow: none !important; background: #FAFAFA }

/* Make the citation column a flex container that can shrink */
#cite_col {
  display: flex !important;
  flex-direction: column !important;
  height: 100vh !important;
  min-height: 0 !important; /* critical so the child can scroll */
}

#citation_md {
  flex: 1 1 auto !important;
  min-height: 0 !important;     /* allow shrinking below content height */
  max-height: 100% !important;  /* cap to the column height */
  overflow: auto !important;    /* show a scrollbar when content is long */
  box-sizing: border-box !important;
  padding: 20px !important;
  background-color: #f9f9f9 !important;
  border-radius: 8px;           /* optional */
}

/* Prevent long words/URLs from overflowing */
#citation_md * {
  overflow-wrap: anywhere !important;
  word-break: break-word !important;
}

#citation_md pre {
  white-space: pre;             /* keep formatting */
  overflow: auto !important;    /* scroll if wide */
  max-width: 100% !important;
}
#citation_md code {
  white-space: pre-wrap;        /* wrap inline code if needed */
}
#citation_md table {
  display: block !important;
  max-width: 100% !important;
  overflow: auto !important;
}
#citation_md img {
  max-width: 100% !important;
  height: auto !important;
}
"""

js=r"""
        () => {
        console.log("📌 Citation-click listener loaded");
        document.addEventListener("click", (e) => {
            const a = e.target.closest('a[href^="#show-cit"]');
            if (!a) return;
            e.preventDefault();

            // parse href fragment
            const href = a.getAttribute("href");
            const qs = href.split("?")[1] || "";
            const params = new URLSearchParams(qs);
            const qid = params.get("qid") || "";

            // find the actual <input> or <textarea> inside the hidden Textbox
            const qidContainer = document.getElementById("qid-hidden");
            const qidBox = qidContainer?.querySelector("input, textarea");

            if (qidBox) {
                qidBox.value = qid;
                // fire an input event so Gradio knows it changed
                qidBox.dispatchEvent(new Event("input", { bubbles: true }));
            }

            // click the hidden button to call Python
            const btn = document.getElementById("cit-trigger");
            if (btn) btn.click();
        });
        }
        """

with gr.Blocks(fill_height=True, title="Tengu Compass", css=css) as iface:
    """
    Sets up the Gradio interface for the chatbot, including text input and buttons.
    """
    with gr.Row(scale=1):
        with gr.Sidebar():
            gr.HTML("""
            <div style="text-align: center;">
                <h1 style="margin: 0 0 10px 0; font-size: 24px; color: #333;">Tengu Compass</h1>
                <p style="color: #666; font-size: 14px;">Smart Document Assistant</p>
                <hr style="margin: 20px 0;">
            </div>
            """)

            conversations_state = gr.State({})
            conv_selector = gr.Radio(
                label="会話一覧",
                choices=[],
                interactive=True,
                elem_id="gradio-radio",
            )

        with gr.Column(scale=7):
            chatbot = gr.Chatbot(
                [],
                type="messages",
                show_copy_button=True,
                avatar_images=(None, "/home/app/assets/robot-face_1f916.png"),
                elem_id="chatbot",
                show_label=False
            )

            with gr.Column(elem_id="input_container"):
                msg = gr.Textbox(
                    show_label=False,
                    container=False,
                    autofocus=True,
                    placeholder="ここに質問を入力してください...",
                    elem_id="textbox",
                )
                with gr.Row():
                    submit_btn = gr.Button("送信", variant="primary")
                    clear = gr.Button("クリア")

        with gr.Column(scale=5, elem_id="cite_col"):
            citation_display = gr.Markdown(
                "# 参照\n\n参照を選択してください。",
                elem_id="citation_md"
            )

            gr.HTML("""
            <style>
                #citation_md {
                    padding: 20px;
                    background-color: #f9f9f9;
                }
            </style>
            """)

    # Hidden components to bridge between JS and Python for citation links
    qid_hidden = gr.Textbox(visible=False, elem_id="qid-hidden")
    cit_trigger = gr.Button(visible=False, elem_id="cit-trigger")

    # Wire hidden trigger to handler
    cit_trigger.click(
        handle_citation_link_click,
        inputs=[qid_hidden],
        outputs=[citation_display],
    )

    txt_msg = msg.submit(user, [msg, chatbot], [msg, chatbot]).then(
        respond, [msg, chatbot], chatbot
    )
    submit_btn.click(user, [msg, chatbot], [msg, chatbot]).then(
        respond, [msg, chatbot], chatbot
    )
    chatbot.like(handle_like, None, None)
    chatbot.retry(handle_retry, [chatbot], [chatbot])

    # Clear: reset chat, then refresh sidebar to include just-finished conversation, then clear citation
    clear.click(clear_chat, None, [chatbot]).then(
        refresh_conv_selector, None, [conv_selector, conversations_state]
    ).then(
        refresh_citation_display, None, [citation_display]
    )

    # Populate sessions list from Langfuse on load
    # Define the handler for clicks on citation links
    iface.load(
        fn=refresh_conv_selector,
        inputs=[],
        outputs=[conv_selector, conversations_state],
        js=js
    )

    # Load selected session into Chatbot (and route subsequent turns to that session id)
    conv_selector.change(
        on_select_conv,
        inputs=[conv_selector, conversations_state],
        outputs=[chatbot]
    )

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=8080, show_error=True, favicon_path="/home/app/assets/tengu_compass_logo.png")
