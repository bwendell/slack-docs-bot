from src.retrieval.query_engine import QueryResult

def format_response(result: QueryResult) -> str:
    """Format a query result for Slack."""
    lines = [result.answer, "", "---", "*Sources:*"]

    for i, source in enumerate(result.sources[:3], 1):  # Top 3 sources
        source_link = source.source_path
        if source.source_type == 'docs' and source_link.startswith('http'):
            lines.append(f"{i}. <{source_link}|{source_link}>")
        else:
            lines.append(f"{i}. `{source_link}`")

    return "\n".join(lines)
