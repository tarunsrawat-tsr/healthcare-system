import typer
from rich import print
from .pipeline import build_pipeline

app = typer.Typer()

@app.command()
def report(source: str, top: int = 15):
    """Run pipeline and show top anomalies"""
    print(f"[bold]Scoring {source}...[/bold]")
    con = build_pipeline(source)
    
    results = con.execute(f"""
        SELECT provider_id, drg, ROUND(markup_ratio, 2) as markup, ROUND(robust_z, 1) as z_score 
        FROM final_results 
        WHERE robust_z >= 3.5 
        ORDER BY robust_z DESC LIMIT {top}
    """).df()
    
    print(results)

if __name__ == "__main__":
    app()