#!/usr/bin/env python
"""
Monitoring script for HPI-FHFA pipeline
Provides real-time monitoring and visualization of pipeline metrics
"""

import argparse
import json
import time
import sys
from datetime import datetime
from typing import Dict, List
import requests
from rich.console import Console
from rich.table import Table
from rich.layout import Layout
from rich.panel import Panel
from rich.live import Live
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn


console = Console()


class PipelineMonitor:
    """Monitor for HPI-FHFA pipeline execution"""
    
    def __init__(self, spark_ui_url: str = "http://localhost:4040", 
                 metrics_endpoint: str = None):
        self.spark_ui_url = spark_ui_url
        self.metrics_endpoint = metrics_endpoint
        self.metrics_history = []
        
    def get_spark_metrics(self) -> Dict:
        """Fetch metrics from Spark UI API"""
        try:
            # Get application info
            app_response = requests.get(f"{self.spark_ui_url}/api/v1/applications")
            if app_response.status_code == 200:
                apps = app_response.json()
                if apps:
                    app_id = apps[0]["id"]
                    
                    # Get job info
                    jobs_response = requests.get(
                        f"{self.spark_ui_url}/api/v1/applications/{app_id}/jobs"
                    )
                    jobs = jobs_response.json() if jobs_response.status_code == 200 else []
                    
                    # Get executor info
                    executors_response = requests.get(
                        f"{self.spark_ui_url}/api/v1/applications/{app_id}/executors"
                    )
                    executors = executors_response.json() if executors_response.status_code == 200 else []
                    
                    return {
                        "application": apps[0],
                        "jobs": jobs,
                        "executors": executors
                    }
        except Exception as e:
            console.print(f"[red]Failed to fetch Spark metrics: {e}[/red]")
        
        return None
    
    def get_custom_metrics(self) -> Dict:
        """Fetch custom pipeline metrics"""
        if self.metrics_endpoint:
            try:
                response = requests.get(self.metrics_endpoint)
                if response.status_code == 200:
                    return response.json()
            except Exception as e:
                console.print(f"[yellow]Failed to fetch custom metrics: {e}[/yellow]")
        
        return None
    
    def create_dashboard(self) -> Layout:
        """Create monitoring dashboard layout"""
        layout = Layout()
        
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=3)
        )
        
        layout["body"].split_row(
            Layout(name="left"),
            Layout(name="right")
        )
        
        # Header
        layout["header"].update(
            Panel("🏠 HPI-FHFA Pipeline Monitor", style="bold blue")
        )
        
        # Footer
        layout["footer"].update(
            Panel(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                  style="dim")
        )
        
        return layout
    
    def update_spark_info(self, layout: Layout, metrics: Dict):
        """Update Spark information panel"""
        if not metrics:
            layout["left"].update(Panel("No Spark metrics available", title="Spark Status"))
            return
            
        # Create Spark info table
        table = Table(title="Spark Application")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")
        
        app = metrics.get("application", {})
        table.add_row("Application ID", app.get("id", "N/A"))
        table.add_row("Name", app.get("name", "N/A"))
        table.add_row("Start Time", app.get("attempts", [{}])[0].get("startTime", "N/A"))
        
        # Jobs summary
        jobs = metrics.get("jobs", [])
        active_jobs = [j for j in jobs if j.get("status") == "RUNNING"]
        completed_jobs = [j for j in jobs if j.get("status") == "SUCCEEDED"]
        failed_jobs = [j for j in jobs if j.get("status") == "FAILED"]
        
        table.add_row("Active Jobs", str(len(active_jobs)))
        table.add_row("Completed Jobs", str(len(completed_jobs)))
        table.add_row("Failed Jobs", str(len(failed_jobs)))
        
        # Executor summary
        executors = metrics.get("executors", [])
        active_executors = [e for e in executors if e.get("isActive", False)]
        
        table.add_row("Active Executors", str(len(active_executors)))
        
        if active_executors:
            total_cores = sum(e.get("totalCores", 0) for e in active_executors)
            total_memory = sum(e.get("maxMemory", 0) for e in active_executors) / (1024**3)
            
            table.add_row("Total Cores", str(total_cores))
            table.add_row("Total Memory", f"{total_memory:.1f} GB")
        
        layout["left"].update(Panel(table, title="Spark Status"))
    
    def update_pipeline_metrics(self, layout: Layout, metrics: Dict):
        """Update pipeline metrics panel"""
        if not metrics:
            layout["right"].update(Panel("No pipeline metrics available", 
                                       title="Pipeline Metrics"))
            return
            
        # Create metrics table
        table = Table(title="Pipeline Progress")
        table.add_column("Stage", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Duration", style="yellow")
        
        stages = metrics.get("stages", {})
        for stage_name, stage_info in stages.items():
            status = stage_info.get("status", "pending")
            duration = stage_info.get("duration_seconds", 0)
            
            status_icon = {
                "completed": "✅",
                "running": "🔄",
                "failed": "❌",
                "pending": "⏳"
            }.get(status, "❓")
            
            table.add_row(
                stage_name,
                f"{status_icon} {status}",
                f"{duration:.1f}s" if duration > 0 else "-"
            )
        
        # Add summary info
        summary = metrics.get("summary", {})
        if summary:
            table.add_section()
            table.add_row("Total Duration", "", f"{summary.get('pipeline_duration_seconds', 0):.1f}s")
            table.add_row("Data Quality Checks", "", 
                         f"{summary.get('data_quality_checks_passed', 0)} passed")
        
        layout["right"].update(Panel(table, title="Pipeline Metrics"))
    
    def monitor_live(self, refresh_interval: int = 5):
        """Live monitoring with auto-refresh"""
        layout = self.create_dashboard()
        
        with Live(layout, refresh_per_second=1, console=console) as live:
            while True:
                try:
                    # Fetch metrics
                    spark_metrics = self.get_spark_metrics()
                    pipeline_metrics = self.get_custom_metrics()
                    
                    # Update dashboard
                    self.update_spark_info(layout, spark_metrics)
                    self.update_pipeline_metrics(layout, pipeline_metrics)
                    
                    # Update footer with timestamp
                    layout["footer"].update(
                        Panel(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
                              f"Press Ctrl+C to exit", style="dim")
                    )
                    
                    time.sleep(refresh_interval)
                    
                except KeyboardInterrupt:
                    console.print("\n[yellow]Monitoring stopped by user[/yellow]")
                    break
                except Exception as e:
                    console.print(f"[red]Error during monitoring: {e}[/red]")
                    time.sleep(refresh_interval)


def main():
    parser = argparse.ArgumentParser(description="Monitor HPI-FHFA Pipeline")
    parser.add_argument(
        "--spark-ui",
        default="http://localhost:4040",
        help="Spark UI URL"
    )
    parser.add_argument(
        "--metrics-endpoint",
        help="Custom metrics endpoint URL"
    )
    parser.add_argument(
        "--refresh",
        type=int,
        default=5,
        help="Refresh interval in seconds"
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run once and exit (no live monitoring)"
    )
    
    args = parser.parse_args()
    
    monitor = PipelineMonitor(
        spark_ui_url=args.spark_ui,
        metrics_endpoint=args.metrics_endpoint
    )
    
    if args.once:
        # Single run
        spark_metrics = monitor.get_spark_metrics()
        pipeline_metrics = monitor.get_custom_metrics()
        
        console.print(Panel("HPI-FHFA Pipeline Status", style="bold blue"))
        
        if spark_metrics:
            console.print("\n[bold]Spark Metrics:[/bold]")
            console.print(spark_metrics)
            
        if pipeline_metrics:
            console.print("\n[bold]Pipeline Metrics:[/bold]")
            console.print(pipeline_metrics)
    else:
        # Live monitoring
        monitor.monitor_live(refresh_interval=args.refresh)


if __name__ == "__main__":
    main()