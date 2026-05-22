import typer
from rich import print
from pydantic import ValidationError
from .pipeline import build_pipeline, PipelineConfig

app = typer.Typer()

@app.command()
def report(
    source: str, 
    top: int = 15,
    min_support: int = typer.Option(8, help="Minimum peer group size"),
    z_threshold: float = typer.Option(3.5, help="Z-Score anomaly threshold")
):
    """Run pipeline and show top anomalies"""
    print(f"[bold]Scoring {source}...[/bold]")
    
    try:
        config = PipelineConfig(
            csv_path=source,
            min_support=min_support,
            z_threshold=z_threshold
        )
    except ValidationError as e:
        print(f"[bold red]Configuration Error:[/bold red]\n{e}")
        raise typer.Exit(code=1)

    con = build_pipeline(config)
    
    results = con.execute(f"""
        SELECT provider_id, drg, ROUND(markup_ratio, 2) as markup, ROUND(robust_z, 1) as z_score 
        FROM final_results 
        WHERE robust_z >= {config.z_threshold} 
        ORDER BY robust_z DESC LIMIT {top}
    """).df()
    
    print(results)

if __name__ == "__main__":
    app()