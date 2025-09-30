from dotenv import load_dotenv
import gradio as gr
import os
from chatbot import ChatBot
from vectordb import PrepareVectorDB
from load_config import load_config_from_yaml 

load_dotenv()  

# Load configuration from YAML file
#config_path = os.path.join('..', 'configs', 'config.yaml')
config = load_config_from_yaml()
directories = config['directories']
retrieval_config = config['retrieval_config']
llm_config = config['llm_config']
embedding_engine = config['embedding_engine']
splitter_config = config['splitter_config']

chatbot_instance = ChatBot()

def update_log_display():
    """Function to update the log display"""
    return chatbot_instance.get_log()

def clear_all():
    """Clear both chat and logs"""
    chatbot_instance.clear_log()
    return "", [], ""  # Clear input, chatbot, and log display

def toggle_sidebar(state):
    """
    Toggle the visibility state of a UI component.
    """
    state = not state
    return gr.update(visible=state), state

def initialize_vectordb():
    """
    Initialize the VectorDB and return status message.
    """
    try:
        # Create the persist directory if it doesn't exist
        os.makedirs(directories['persist_directory'], exist_ok=True)

        prepare_vectordb_instance = PrepareVectorDB(
           # data_directory=directories['data_directory'],
            url=directories['url'],
            persist_directory=directories['persist_directory'],
            embedding_model_engine=embedding_engine['embedding_model_engine'],
            chunk_size=splitter_config['chunk_size'],
            chunk_overlap=splitter_config['chunk_overlap']
        )

        # Check if VectorDB exists by looking for files in the directory
        if len(os.listdir(directories['persist_directory'])) == 0:
            # Directory is empty, create VectorDB
            vectordb = prepare_vectordb_instance.prepare_and_save_vectordb()  ###
            status_message = f"VectorDB successfully created in {directories['persist_directory']}\n"
            status_message += f"Number of documents processed: Check logs for details"
            # Add to chatbot log as well
            chatbot_instance.add_log(f"VectorDB initialized: {status_message}")
        else:
            status_message = f"ℹVectorDB already exists in {directories['persist_directory']}"
            chatbot_instance.add_log(status_message)

        return status_message

    except Exception as e:
        error_message = f"Error initializing VectorDB: {str(e)}"
        chatbot_instance.add_log(error_message)
        return error_message

with gr.Blocks() as demo:
    with gr.Tabs():
        with gr.TabItem("RAG-GPT"):
            ##############
            # First ROW:
            ##############
            with gr.Row() as row_one:
                with gr.Column(visible=False) as reference_bar:
                    ref_output = gr.Markdown()

                with gr.Column() as chatbot_output:
                    chatbot = gr.Chatbot(
                        [],
                        elem_id="chatbot",
                        bubble_full_width=False,
                        height=500,
                    )

            ##############
            # SECOND ROW:
            ##############
            with gr.Row():
                input_txt = gr.Textbox(
                    lines=4,
                    scale=8,
                    placeholder="Enter text and press enter",
                    container=False,
                )

            ##############
            # Third ROW: Buttons
            ##############
            with gr.Row() as row_two:
                with gr.Column(scale=1):
                    init_db_btn = gr.Button(value="Initialize VectorDB", variant="primary")

                with gr.Column(scale=1):
                    text_submit_btn = gr.Button(value="Submit text")

                with gr.Column(scale=1):
                    btn_toggle_sidebar = gr.Button(value="References")

                with gr.Column(scale=1):
                    clear_button = gr.Button(value="Clear All")

                sidebar_state = gr.State(False)

            ##############
            # Fourth ROW: Status and Logs
            ##############
            with gr.Row() as row_three:
                with gr.Column(scale=1):
                    status_display = gr.Textbox(
                        label="VectorDB Status",
                        interactive=False,
                        lines=3,
                        placeholder="VectorDB not initialized yet"
                    )

                with gr.Column(scale=2):
                    log_display = gr.Textbox(
                        label="Logging",
                        interactive=False,
                        lines=10
                    )

            ##############
            # Event Handlers:
            ##############

            # Initialize VectorDB button
            init_db_btn.click(
                lambda: gr.Button(interactive=False, value="Initializing..."),
                None,
                [init_db_btn],
                queue=False
            ).then(
                fn=initialize_vectordb,
                inputs=None,
                outputs=[status_display],
                queue=False
            ).then(
                lambda: gr.Button(interactive=True, value="Initialize VectorDB"),
                None,
                [init_db_btn],
                queue=False
            ).then(
                update_log_display,
                None,
                [log_display],
                queue=False
            )

            # Toggle sidebar
            btn_toggle_sidebar.click(
                toggle_sidebar,
                [sidebar_state],
                [reference_bar, sidebar_state]
            )

            # Clear all
            clear_button.click(
                fn=clear_all,
                inputs=None,
                outputs=[input_txt, chatbot, log_display]
            ).then(
                lambda: "VectorDB status will update on next initialization",
                None,
                [status_display],
                queue=False
            )

            # Text input submit
            txt_msg = input_txt.submit(
                fn=chatbot_instance.respond,
                inputs=[chatbot, input_txt],
                outputs=[input_txt, chatbot, ref_output],
                queue=False
            ).then(
                update_log_display,
                None,
                [log_display],
                queue=False
            ).then(
                lambda: gr.Textbox(interactive=True),
                None,
                [input_txt],
                queue=False
            )

            # Submit button click
            txt_msg = text_submit_btn.click(
                fn=chatbot_instance.respond,
                inputs=[chatbot, input_txt],
                outputs=[input_txt, chatbot, ref_output],
                queue=False
            ).then(
                update_log_display,
                None,
                [log_display],
                queue=False
            ).then(
                lambda: gr.Textbox(interactive=True),
                None,
                [input_txt],
                queue=False
            )


if __name__ == "__main__":
    demo.launch(share=True)