import os
import sys
from collections import deque
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from root .env file
load_dotenv()

class StreamingChatbot:
    """
    A stateful LLM chatbot featuring streaming tokens and sliding window memory management.
    """
    def __init__(self, model: str = "gpt-4o-mini", max_history: int = 6):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is missing in .env")
            
        self.client = OpenAI(api_key=api_key)
        self.model = model
        # Limit stored conversation turns to manage total token context size
        self.history = deque(maxlen=max_history)
        self.system_prompt = {
            "role": "system",
            "content": "You are a helpful, expert AI engineering assistant. Provide clear, concise answers."
        }

    def send_message(self, user_input: str) -> None:
        """
        Sends user message to LLM API, streams output chunks in real time, and logs conversation turn.
        """
        # Append latest user input to sliding window memory
        self.history.append({"role": "user", "content": user_input})
        
        # Build prompt payload: System instructions + limited sliding window turn history
        messages = [self.system_prompt] + list(self.history)
        
        try:
            # Request token streaming completion
            response_stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.3,
                stream=True
            )
            
            print("\nAssistant: ", end="", flush=True)
            accumulated_text = []
            
            # Stream incoming token chunks directly to stdout
            for chunk in response_stream:
                content = chunk.choices[0].delta.content or ""
                print(content, end="", flush=True)
                accumulated_text.append(content)
            print("\n")
            
            # Save full assistant response back to conversation history
            full_response = "".join(accumulated_text)
            self.history.append({"role": "assistant", "content": full_response})

        except Exception as e:
            print(f"\n[Error communicating with LLM API]: {e}\n", file=sys.stderr)

    def run_cli(self) -> None:
        """
        Launches interactive CLI loop for real-time terminal chatting.
        """
        print("==================================================")
        print(f" Stateful Chatbot Engine Initialized ({self.model}) ")
        print(" Type 'exit', 'quit', or 'q' to end the session.   ")
        print("==================================================\n")
        
        while True:
            try:
                user_input = input("User: ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ["exit", "quit", "q"]:
                    print("\nSession closed. Goodbye!")
                    break
                
                self.send_message(user_input)
                
            except (KeyboardInterrupt, EOFError):
                print("\n\nSession terminated by user.")
                break


if __name__ == "__main__":
    chatbot = StreamingChatbot()
    chatbot.run_cli()
