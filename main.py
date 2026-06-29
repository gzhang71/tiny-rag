"""
tiny_rag CLI

Usage:
    python main.py --file path/to/doc.txt "What is the main topic?"
    python main.py --dir path/to/docs/ "Summarize the key points"
    python main.py --text "Alice loves cats. Bob loves dogs." "Who loves cats?"
"""
import argparse
import sys

from rag import RAGPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Tiny RAG — ask questions over your documents")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--file", metavar="PATH", help="Ingest a single text file")
    source.add_argument("--dir", metavar="DIR", help="Ingest all .txt files in a directory")
    source.add_argument("--text", metavar="TEXT", help="Ingest raw text inline")
    parser.add_argument("question", help="Question to answer")
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve (default: 5)")
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=64)
    args = parser.parse_args()

    pipeline = RAGPipeline(chunk_size=args.chunk_size, overlap=args.overlap, top_k=args.top_k)

    if args.file:
        n = pipeline.ingest_file(args.file)
        print(f"Ingested {n} chunks from {args.file}", file=sys.stderr)
    elif args.dir:
        n = pipeline.ingest_directory(args.dir)
        print(f"Ingested {n} chunks from {args.dir}", file=sys.stderr)
    else:
        n = pipeline.ingest_text(args.text, source="cli-inline")
        print(f"Ingested {n} chunks from inline text", file=sys.stderr)

    answer = pipeline.query(args.question)
    print(answer)


if __name__ == "__main__":
    main()
