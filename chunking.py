from langchain_text_splitters import RecursiveCharacterTextSplitter


def make_text_splitter(
    strategy: str = "recursive",
    chunk_size: int = 1000,
    overlap: int = 200,
) -> RecursiveCharacterTextSplitter:
    safe_overlap = min(overlap, chunk_size - 1)

    if strategy == "recursive":
        separators = ["\n\n", "\n", ". ", " ", ""]
    elif strategy == "paragraph":
        separators = ["\n\n", "\n", ""]
    elif strategy == "fixed":
        separators = [""]
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy}")

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=safe_overlap,
        length_function=len,
        separators=separators,
        is_separator_regex=False,
    )


def chunk_text(
    text: str,
    strategy: str = "recursive",
    chunk_size: int = 1000,
    overlap: int = 200,
) -> list[str]:
    splitter = make_text_splitter(
        strategy=strategy,
        chunk_size=chunk_size,
        overlap=overlap,
    )
    return splitter.split_text(text)
