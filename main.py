"""
tiny_rag CLI

Usage:
    python main.py --file path/to/doc.txt "What is the main topic?"
    python main.py --dir path/to/docs/ "Summarize the key points"
    python main.py --text "Alice loves cats. Bob loves dogs." "Who loves cats?"

    # persistent vector DB (Chroma): ingest once, then query without a source
    python main.py --store chroma --file path/to/doc.txt "What is the main topic?"
    python main.py --store chroma "What else does it say?"
"""
import argparse
import sys

from rag import RAGPipeline, StoreBackend


def main() -> None:
    parser = argparse.ArgumentParser(description="Tiny RAG — ask questions over your documents")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--file", metavar="PATH", help="Ingest a single text file")
    source.add_argument("--dir", metavar="DIR", help="Ingest all .txt files in a directory")
    source.add_argument("--text", metavar="TEXT", help="Ingest raw text inline")
    parser.add_argument("question", help="Question to answer")
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve (default: 5)")
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument(
        "--store", choices=[b.value for b in StoreBackend], default=StoreBackend.FAISS.value,
        help="Vector store backend: faiss (in-memory) or chroma (persistent DB) (default: faiss)",
    )
    parser.add_argument(
        "--persist-dir", default="./chroma_db",
        help="Directory for the Chroma DB when --store chroma (default: ./chroma_db)",
    )
    parser.add_argument(
        "--chroma-host", default=None,
        help="Connect to a running Chroma server (`chroma run`) instead of the embedded DB",
    )
    parser.add_argument("--chroma-port", type=int, default=8000)
    args = parser.parse_args()

    pipeline = RAGPipeline(
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        top_k=args.top_k,
        backend=StoreBackend(args.store),
        persist_dir=args.persist_dir,
        chroma_host=args.chroma_host,
        chroma_port=args.chroma_port,
    )

    if args.file:
        n = pipeline.ingest_file(args.file)
        print(f"Ingested {n} chunks from {args.file}", file=sys.stderr)
    elif args.dir:
        n = pipeline.ingest_directory(args.dir)
        print(f"Ingested {n} chunks from {args.dir}", file=sys.stderr)
    elif args.text:
        n = pipeline.ingest_text(args.text, source="cli-inline")
        print(f"Ingested {n} chunks from inline text", file=sys.stderr)
    elif len(pipeline.store) == 0:
        parser.error("the store is empty — provide a source (--file, --dir, or --text)")
    else:
        print(f"Querying existing store ({len(pipeline.store)} chunks)", file=sys.stderr)

    answer = pipeline.query(args.question)
    print(answer)


if __name__ == "__main__":
    main()
