"""
Main Driver Script
Unified entry point for all chat modes (Terminal, API, UI)
"""
import argparse
import sys


def main():
    """Main driver function"""
    
    # Setup argument parser
    parser = argparse.ArgumentParser(
        description="Mental Health RAG Chatbot - Unified Driver",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run web UI
  python driver.py --mode ui
  
  # Run offline indexing
  python driver.py --mode index

Chat Modes:
  terminal  - Interactive terminal-based chat (VSCode terminal)
  api       - REST API server (Flask)
  ui        - Web-based UI (Gradio)
  index     - Build/rebuild knowledge base (one-time setup)
        """
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['ui', 'index'],
        help='Chat mode to run'
    )
    
    parser.add_argument(
        '--lightweight',
        action='store_true',
        help='Use lightweight crisis classifier (less memory)'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        help='Port for API/UI server (overrides config)'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Display header
    print("\n" + "=" * 80)
    print("MENTAL HEALTH RAG CHATBOT - DRIVER")
    print("=" * 80)
    print(f"Mode: {args.mode.upper()}")
    if args.lightweight:
        print("Using: Lightweight crisis classifier")
    print("=" * 80 + "\n")
    
    # Route to appropriate mode
    try:
        if args.mode == 'ui':
            import os

            # Pass lightweight flag via environment variable
            if args.lightweight:
                os.environ["LIGHTWEIGHT_CLASSIFIER"] = "1"

            print("[Driver] Launching Streamlit UI...\n")

            # Launch Streamlit programmatically
            import subprocess
            subprocess.run(["streamlit", "run", "app.py"])
        
        elif args.mode == 'index':
            print("[Driver] Running offline indexing pipeline...")
            print("This will download the dataset and build the knowledge base.")
            print("This may take 10-30 minutes depending on your system.\n")
            
            response = input("Continue? (y/n): ").strip().lower()
            if response == 'y':
                from offline_indexing import main as index_main
                index_main()
            else:
                print("\nIndexing cancelled.")
                sys.exit(0)
    
    except KeyboardInterrupt:
        print("\n\n[Driver] Interrupted by user. Exiting...")
        sys.exit(0)
    
    except Exception as e:
        print(f"\n[Driver] Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
