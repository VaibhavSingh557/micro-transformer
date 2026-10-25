"""Train the Micro-Transformer and serve the attention dashboard."""
import argparse
import threading
import webbrowser

from micro_transformer import Config, MicroTransformer
from micro_transformer.server import make_server

CORPUS = [
    "neural attention layers transform input tokens into context vectors .",
    "gradient descent minimizes cross entropy loss across training epochs .",
    "causal masking blocks future tokens during autoregressive text generation .",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and serve the Micro-Transformer attention explorer.")
    parser.add_argument("--host", default="127.0.0.1", help="interface to bind (default: localhost only)")
    parser.add_argument("--port", type=int, default=5095)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    args = parser.parse_args()

    engine = MicroTransformer(CORPUS, Config())
    losses = engine.train(CORPUS, epochs=args.epochs, lr=0.01)
    sample = engine.generate("neural attention layers", max_new_tokens=6)
    print(f"[+] Trained {args.epochs} epochs | loss {losses[0]:.3f} -> {losses[-1]:.3f}")
    print(f"[+] Sample completion: 'neural attention layers {sample['completion']}'")

    server = make_server(engine, args.host, args.port)
    url = f"http://{args.host}:{args.port}"
    print(f"[+] Dashboard running at {url} (Ctrl+C to stop)")
    if not args.no_browser:
        threading.Timer(1.0, webbrowser.open, args=(url,)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[*] Shutting down")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
