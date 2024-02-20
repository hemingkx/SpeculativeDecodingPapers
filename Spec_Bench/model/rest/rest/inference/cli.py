# Adapted from: https://github.com/lm-sys/FastChat/blob/main/fastchat/serve/cli.py
# Adapted from: https://github.com/FasterDecoding/Medusa/blob/main/medusa/inference/cli.py
"""
Chat with a model with command line interface.

Usage:
python3 -m rest.inference.cli --datastore-path <> --base-model <>
Other commands:
- Type "!!exit" or an empty line to exit.
- Type "!!reset" to start a new conversation.
- Type "!!remove" to remove the last prompt.
- Type "!!regen" to regenerate the last message.
- Type "!!save <filename>" to save the conversation history to a json file.
- Type "!!load <filename>" to load a conversation history from a json file.
"""
import argparse
import os
import re
import sys
import torch
from fastchat.serve.cli import SimpleChatIO, RichChatIO, ProgrammaticChatIO
from fastchat.model.model_adapter import get_conversation_template
from fastchat.conversation import get_conv_template
import json
from rest.model.rest_model import RestModel

import draftretriever


def main(args):
    if args.style == "simple":
        chatio = SimpleChatIO(args.multiline)
    elif args.style == "rich":
        chatio = RichChatIO(args.multiline, args.mouse)
    elif args.style == "programmatic":
        chatio = ProgrammaticChatIO()
    else:
        raise ValueError(f"Invalid style for console: {args.style}")
    try:
        model = RestModel.from_pretrained(
            args.base_model,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto",
            load_in_8bit=args.load_in_8bit,
            load_in_4bit=args.load_in_4bit,
        )
        tokenizer = model.get_tokenizer()
        if not args.baseline:
            datastore = draftretriever.Reader(
                        index_file_path=args.datastore_path,  # "/data/tianle2/hzy/ret_decode/datastore/vicuna_ultrachat.idx",
                    )
        conv = None

        def new_chat():
            return get_conversation_template("vicuna")

        def reload_conv(conv):
            """
            Reprints the conversation from the start.
            """
            for message in conv.messages[conv.offset :]:
                chatio.prompt_for_output(message[0])
                chatio.print_output(message[1])

        while True:
            if not conv:
                conv = new_chat()

            try:
                inp = chatio.prompt_for_input(conv.roles[0])
            except EOFError:
                inp = ""

            if inp == "!!exit" or not inp:
                print("exit...")
                break
            elif inp == "!!reset":
                print("resetting...")
                conv = new_chat()
                continue
            elif inp == "!!remove":
                print("removing last message...")
                if len(conv.messages) > conv.offset:
                    # Assistant
                    if conv.messages[-1][0] == conv.roles[1]:
                        conv.messages.pop()
                    # User
                    if conv.messages[-1][0] == conv.roles[0]:
                        conv.messages.pop()
                    reload_conv(conv)
                else:
                    print("No messages to remove.")
                continue
            elif inp == "!!regen":
                print("regenerating last message...")
                if len(conv.messages) > conv.offset:
                    # Assistant
                    if conv.messages[-1][0] == conv.roles[1]:
                        conv.messages.pop()
                    # User
                    if conv.messages[-1][0] == conv.roles[0]:
                        reload_conv(conv)
                        # Set inp to previous message
                        inp = conv.messages.pop()[1]
                    else:
                        # Shouldn't happen in normal circumstances
                        print("No user message to regenerate from.")
                        continue
                else:
                    print("No messages to regenerate.")
                    continue
            elif inp.startswith("!!save"):
                args = inp.split(" ", 1)

                if len(args) != 2:
                    print("usage: !!save <filename>")
                    continue
                else:
                    filename = args[1]

                # Add .json if extension not present
                if not "." in filename:
                    filename += ".json"

                print("saving...", filename)
                with open(filename, "w") as outfile:
                    json.dump(conv.dict(), outfile)
                continue
            elif inp.startswith("!!load"):
                args = inp.split(" ", 1)

                if len(args) != 2:
                    print("usage: !!load <filename>")
                    continue
                else:
                    filename = args[1]

                # Check if file exists and add .json if needed
                if not os.path.exists(filename):
                    if (not filename.endswith(".json")) and os.path.exists(
                        filename + ".json"
                    ):
                        filename += ".json"
                    else:
                        print("file not found:", filename)
                        continue

                print("loading...", filename)
                with open(filename, "r") as infile:
                    new_conv = json.load(infile)

                conv = get_conv_template(new_conv["template_name"])
                conv.set_system_message(new_conv["system_message"])
                conv.messages = new_conv["messages"]
                reload_conv(conv)
                continue

            conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            try:
                chatio.prompt_for_output(conv.roles[1])
                input_ids = tokenizer.encode(prompt, return_tensors="pt").to(
                    model.base_model.device
                )
                outputs = chatio.stream_output(
                    model.rest_generate(
                        input_ids,
                        datastore,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        max_steps=args.max_steps,
                    ) if not args.baseline else \
                    model.baseline_generate(
                        input_ids,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        max_steps=args.max_steps,
                    )
                )
                conv.update_last_message(outputs.strip())

            except KeyboardInterrupt:
                print("stopped generation.")
                # If generation didn't finish
                if conv.messages[-1][1] is None:
                    conv.messages.pop()
                    # Remove last user message, so there isn't a double up
                    if conv.messages[-1][0] == conv.roles[0]:
                        conv.messages.pop()

                    reload_conv(conv)

    except KeyboardInterrupt:
        print("exit...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datastore-path", type=str, required=True, help="datastore path.")
    parser.add_argument("--base-model", type=str, default="lmsys/vicuna-7b-v1.5", help="LLM name or path.")
    parser.add_argument(
        "--load-in-8bit", action="store_true", help="Use 8-bit quantization"
    )
    parser.add_argument(
        "--load-in-4bit", action="store_true", help="Use 4-bit quantization"
    )
    parser.add_argument(
        "--conv-template", type=str, default=None, help="Conversation prompt template."
    )
    parser.add_argument(
        "--conv-system-msg", type=str, default=None, help="Conversation system message."
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--max-steps", type=int, default=512)
    parser.add_argument("--baseline", action="store_true", help="Use standard autoregressive generation.")
    parser.add_argument("--no-history", action="store_true")
    parser.add_argument(
        "--style",
        type=str,
        default="simple",
        choices=["simple", "rich", "programmatic"],
        help="Display style.",
    )
    parser.add_argument(
        "--multiline",
        action="store_true",
        help="Enable multiline input. Use ESC+Enter for newline.",
    )
    parser.add_argument(
        "--mouse",
        action="store_true",
        help="[Rich Style]: Enable mouse support for cursor positioning.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print useful debug information (e.g., prompts)",
    )
    args = parser.parse_args()
    main(args)
