# TUI Application - Management Console with ML Pipeline
import asyncio
import subprocess
from pathlib import Path
from textual.app import App
from textual.containers import Horizontal, Vertical
from textual.widgets import Header, Footer, Button, DataTable, Label, ListView, ListItem, RichLog, Select
from textual.binding import Binding
from textual import work

from .config import (
    COMPONENTS, ML_PIPELINE, SPOOF_ATTACKS,
    GZ_WORLDS, DEFAULT_GZ_WORLD,
    RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR,
    PROJECT_ROOT,
    get_terrain_model_status,
)
from .process_manager import is_process_running, get_pid
from .widgets import (
    start_component,
    stop_component,
    stop_gazebo,
    is_gazebo_running,
    stop_all_components,
    run_ml_pipeline,
    get_data_info,
)


class GPSDetectorApp(App):
    """GPS Spoofing Detection — Management Console."""

    TITLE = "GPS Spoofing Detection - Management Console"

    CSS = """
    /* ── Global ─────────────────────────────────────────────── */
    Screen {
        background: $background;
    }
    Horizontal {
        height: auto;
    }
    .component-btn {
        width: 100%;
        margin: 1 0;
    }
    #spoof-panel Button {
        width: 1fr;
        margin: 0 1;
    }
    Label.section-title {
        text-style: bold;
        margin-bottom: 1;
    }
    Label.separator {
        height: 1;
    }

    /* ── Row 1 ───────────────────────────────────────────────── */
    #left-panel {
        width: 28%;
        border: solid green;
        padding: 1;
        overflow-y: auto;
    }
    #world-select {
        width: 100%;
        margin-bottom: 1;
    }
    #world-terrain-hint {
        margin-bottom: 1;
    }
    #middle-panel {
        width: 22%;
        border: solid yellow;
        padding: 1;
        overflow-y: auto;
    }
    #right-panel {
        width: 50%;
        border: solid blue;
        padding: 1;
        overflow-y: auto;
    }
    #status-table {
        height: auto;
        overflow-y: auto;
    }
    #log-viewer {
        height: 12;
        border: solid $panel;
        overflow-y: auto;
        margin-top: 1;
    }

    /* ── Row 2 ───────────────────────────────────────────────── */
    #data-panel {
        width: 50%;
        border: solid purple;
        padding: 1;
        overflow-y: auto;
    }
    #ml-panel {
        width: 50%;
        border: solid orange;
        padding: 1;
        overflow-y: auto;
    }

    /* ── Row 3 ───────────────────────────────────────────────── */
    #models-panel {
        width: 35%;
        border: solid cyan;
        padding: 1;
        overflow-y: auto;
    }
    #terrain-panel {
        width: 65%;
        border: solid magenta;
        padding: 1;
        overflow-y: auto;
    }
    #terrain-table {
        height: auto;
        overflow-y: auto;
    }
    /* ── Row 4 ─────────────────────────────────────────────── */
    #spoof-panel {
        width: 100%;
        border: solid red;
        padding: 1;
    }
    #spoof-log {
        height: 1fr;
        border: solid $panel;
        margin-top: 1;
    }
    """

    # ── Auto-refresh timer ────────────────────────────────────────
    AUTO_REFRESH_INTERVAL = 5.0   # seconds
    LOG_TAIL_INTERVAL = 0.5       # seconds

    def compose(self):
        yield Header(show_clock=True)

        # ── ROW 1: Components / Actions / Status+Log ──────────────
        with Horizontal():
            with Vertical(id="left-panel"):
                yield Label("[b]🌍 Simulation World[/b]")
                yield Select(
                    options=[(cfg["label"], world_id) for world_id, cfg in GZ_WORLDS.items()],
                    value=DEFAULT_GZ_WORLD,
                    id="world-select",
                )
                yield Label(f"[dim]Terrain model: {GZ_WORLDS[DEFAULT_GZ_WORLD]['terrain']}[/dim]", id="world-terrain-hint")
                yield Label("", classes="separator")
                yield Label("[b]Components[/b]")
                yield ListView(id="component-list")

            with Vertical(id="middle-panel"):
                yield Label("[b]Actions[/b]")
                yield Button("▶  Start",    id="start",     variant="success", classes="component-btn")
                yield Button("■  Stop",     id="stop",      variant="error",   classes="component-btn")
                yield Label("", classes="separator")
                yield Button("▶▶ Start All", id="start-all", variant="success", classes="component-btn")
                yield Button("■■ Stop All",  id="stop-all",  variant="error",   classes="component-btn")
                yield Button("↺  Refresh",   id="refresh",   variant="primary", classes="component-btn")
                yield Label("", classes="separator")
                yield Button("✕  Kill Gazebo", id="kill-gazebo", variant="warning", classes="component-btn")

            with Vertical(id="right-panel"):
                yield Label("[b]Status[/b]")
                yield DataTable(id="status-table")
                yield Label("[b]Log[/b]")
                yield RichLog(id="log-viewer", markup=True, highlight=True)

        # ── ROW 2: Data Files / ML Pipeline ──────────────────────
        with Horizontal():
            with Vertical(id="data-panel"):
                yield Label("[b]Data Files[/b]")
                yield Button("View Raw Files",   id="view-raw",       variant="primary", classes="component-btn")
                yield Button("View Processed",   id="view-processed", variant="primary", classes="component-btn")
                yield ListView(id="file-list")
                yield Label("[b]File Actions[/b]")
                yield Button("Process Selected", id="process-selected", variant="primary", classes="component-btn")
                yield Button("Delete Selected",  id="delete-selected",  variant="error",   classes="component-btn")

            with Vertical(id="ml-panel"):
                yield Label("[b]ML Pipeline — Global[/b]")
                yield Button("Process All Data",  id="ml-process", variant="primary", classes="component-btn")
                yield Button("Train Global Model", id="ml-train",   variant="primary", classes="component-btn")
                yield Button("Full Pipeline",      id="ml-full",    variant="primary", classes="component-btn")
                yield Label("", classes="separator")
                yield Label("[b]ML Pipeline — Terrain[/b]")
                yield Button("Train Terrain Models",  id="ml-train-terrain", variant="success", classes="component-btn")
                yield Button("Full Terrain Pipeline", id="ml-full-terrain",  variant="success", classes="component-btn")

        # ── ROW 3: Global Models / Terrain Models ─────────────────
        with Horizontal():
            with Vertical(id="models-panel"):
                yield Label("[b]Global Models[/b]")
                yield ListView(id="model-list")

            with Vertical(id="terrain-panel"):
                yield Label("[b]Terrain Models[/b]")
                yield DataTable(id="terrain-table")

        # ── ROW 4: GPS Spoof Injector ─────────────────────────────
        with Horizontal():
            with Vertical(id="spoof-panel"):
                yield Label("[b][red]⚡ GPS Spoof Injector[/red][/b]  — inject a live attack to test your detection system")
                with Horizontal():
                    yield Button("🌀  Drift",        id="spoof-drift",    variant="error",   classes="component-btn")
                    yield Button("📈  Ramp",         id="spoof-ramp",     variant="error",   classes="component-btn")
                    yield Button("🚀  Jump+Drift",   id="spoof-jump",     variant="error",   classes="component-btn")
                    yield Button("📍  Freeze",       id="spoof-static",   variant="error",   classes="component-btn")
                    yield Button("📡  Noise",        id="spoof-noise",    variant="warning", classes="component-btn")
                    yield Button("🛸  Teleport",     id="spoof-teleport", variant="error",   classes="component-btn")
                yield RichLog(id="spoof-log", markup=True)

        yield Footer()

    # ── Lifecycle ─────────────────────────────────────────────────

    def on_mount(self):
        self.populate_component_list()
        self.refresh_table()
        self.append_log("[green]System ready.[/green] Select a component then Start / Stop.")
        self.check_gazebo_status()
        self.show_data_info()
        self.populate_model_list()
        self.populate_terrain_table()
        
        # Log tailing state
        self.log_offsets = {}
        
        # Start timers
        self.set_interval(self.AUTO_REFRESH_INTERVAL, self._auto_refresh)
        self.set_interval(self.LOG_TAIL_INTERVAL, self._tail_logs)

    def _auto_refresh(self):
        """Called every AUTO_REFRESH_INTERVAL seconds to keep status current."""
        self.refresh_table()

    def _tail_logs(self):
        """Monitor component log files and append new lines to RichLog."""
        for comp_id, cfg in COMPONENTS.items():
            if cfg.get("spawn_method") != "background":
                continue
            
            log_file = cfg.get("log_file")
            if not log_file:
                continue
                
            log_path = Path(log_file)
            if not log_path.exists():
                continue
            
            # Initialise offset if not present
            if comp_id not in self.log_offsets:
                # Start at the end of existing file to avoid flooding
                self.log_offsets[comp_id] = log_path.stat().st_size
            
            try:
                with open(log_path, "r", errors="replace") as f:
                    f.seek(self.log_offsets[comp_id])
                    lines = f.readlines()
                    if lines:
                        self.log_offsets[comp_id] = f.tell()
                        for line in lines:
                            line = line.strip()
                            if not line:
                                continue
                            
                            # Special formatting for telemetry
                            if "TELEMETRY:" in line:
                                self.append_log(f"[bold cyan]{line}[/bold cyan]")
                            # Special formatting for anomalies
                            elif "ANOMALY" in line or "p(anom)" in line:
                                if "ANOMALY" in line:
                                    self.append_log(f"[bold red]{line}[/bold red]")
                                else:
                                    self.append_log(f"[yellow]{line}[/yellow]")
                            else:
                                # Normal component log output
                                self.append_log(f"[dim]{cfg['name']}: {line}[/dim]")
            except Exception:
                pass

    def on_select_changed(self, event: Select.Changed):
        """Fires whenever the user picks a different world from the dropdown."""
        if event.select.id != "world-select":
            return
        world   = str(event.value)
        cfg     = GZ_WORLDS.get(world, {})
        terrain = cfg.get("terrain", "flat")
        label   = cfg.get("label", world)

        # Update the PX4 launch command to pass the selected world
        COMPONENTS["px4_sim"]["command"] = (
            f"/Users/nikhilsarwara/research/gps_spoofing/start_px4.sh {world}"
        )

        # Update the terrain hint label
        self.query_one("#world-terrain-hint", Label).update(
            f"[dim]Terrain model: [yellow]{terrain}[/yellow][/dim]"
        )

        self.append_log(
            f"[cyan]🌍 World:[/cyan] {world}  "
            f"→ terrain model [yellow]{terrain}[/yellow]  "
            f"| Start PX4 SITL to apply"
        )

    # ── Population helpers ────────────────────────────────────────

    def populate_component_list(self):
        lv = self.query_one("#component-list", ListView)
        lv.clear()
        for comp_id, cfg in COMPONENTS.items():
            lv.append(ListItem(Label(cfg["name"])))

    def refresh_table(self):
        table = self.query_one("#status-table", DataTable)
        if not table.columns:
            table.add_columns("Component", "Status", "PID")
        table.clear()
        for comp_id, cfg in COMPONENTS.items():
            pid     = get_pid(comp_id)
            running = pid and is_process_running(pid)
            status  = "[green]RUNNING[/green]" if running else "[red]STOPPED[/red]"
            table.add_row(cfg["name"], status, str(pid) if pid else "-")
        # Gazebo row
        gz_running = is_gazebo_running()
        gz_pid     = self._get_gazebo_pid()
        gz_status  = "[green]RUNNING[/green]" if gz_running else "[red]STOPPED[/red]"
        table.add_row("Gazebo Simulator", gz_status, str(gz_pid) if gz_pid else "-")

    def _get_gazebo_pid(self):
        try:
            result = subprocess.run(["pgrep", "-af", "gz sim"],
                                    capture_output=True, text=True)
            if result.returncode == 0 and result.stdout.strip():
                return int(result.stdout.strip().split()[0])
        except Exception:
            pass
        return None

    def check_gazebo_status(self):
        if is_gazebo_running():
            self.append_log("[yellow]WARNING:[/yellow] Gazebo is already running.")

    def populate_file_list(self, file_type="raw"):
        lv = self.query_one("#file-list", ListView)
        lv.clear()
        if file_type == "raw":
            data_dir  = Path(str(RAW_DATA_DIR))
            label_str = "Raw"
        else:
            data_dir  = Path(str(PROCESSED_DATA_DIR))
            label_str = "Processed"
        if not data_dir.exists():
            self.append_log(f"[red]{label_str} dir not found[/red]")
            return
        files = sorted(data_dir.glob("*.csv"))
        for f in files:
            lv.append(ListItem(Label(f.name)))
        self.append_log(f"[cyan]{label_str} files:[/cyan] {len(files)} found")

    def show_data_info(self):
        self.populate_file_list("raw")

    def populate_model_list(self):
        lv = self.query_one("#model-list", ListView)
        lv.clear()
        # Show models from BOTH ml/models/ and project/ml/models/
        ml_models_dir      = PROJECT_ROOT / "ml" / "models"
        project_models_dir = Path(str(MODELS_DIR))
        model_files = []
        for d in [ml_models_dir, project_models_dir]:
            if d.exists():
                model_files += sorted(d.glob("*.pkl")) + sorted(d.glob("*.pth"))
        # Deduplicate by name
        seen, unique = set(), []
        for f in model_files:
            if f.name not in seen:
                seen.add(f.name)
                unique.append(f)
        for f in unique:
            lv.append(ListItem(Label(f.name)))
        self.append_log(f"[cyan]Global models:[/cyan] {len(unique)} found")

    def populate_terrain_table(self):
        table = self.query_one("#terrain-table", DataTable)
        table.clear(columns=True)
        table.add_columns("Terrain", "Status", "RF Val F1", "CNN Val F1", "Windows")
        status = get_terrain_model_status()
        if not status:
            table.add_row("No terrain models trained yet", "", "", "", "")
            return
        for terrain, info in status.items():
            rf_f1  = f"{info['rf_val_f1']:.3f}"  if info.get("rf_val_f1")  is not None else "—"
            cnn_f1 = f"{info['cnn_val_f1']:.3f}" if info.get("cnn_val_f1") is not None else "—"
            stat   = info.get("status", "unknown")
            colour = "green" if stat == "trained" else "yellow"
            table.add_row(
                terrain.capitalize(),
                f"[{colour}]{stat}[/{colour}]",
                rf_f1,
                cnn_f1,
                str(info.get("n_windows", "—")),
            )

    def append_log(self, message):
        self.query_one("#log-viewer", RichLog).write(message)

    # ── Button handler ────────────────────────────────────────────

    def on_button_pressed(self, event: Button.Pressed):
        bid = event.button.id

        # ── Component control ──────────────────────────
        if bid == "start":
            self.start_selected()
        elif bid == "stop":
            self.stop_selected()
        elif bid == "start-all":
            for comp_id in COMPONENTS:
                ok, msg = start_component(comp_id)
                self.append_log(msg)
            self.refresh_table()
        elif bid == "stop-all":
            stop_all_components()
            self.refresh_table()
            self.append_log("[red]All components stopped.[/red]")
        elif bid == "refresh":
            self.refresh_table()
            self.show_data_info()
            self.populate_terrain_table()
            self.append_log("[cyan]Refreshed.[/cyan]")
        elif bid == "kill-gazebo":
            stop_gazebo()
            self.append_log("[red]Gazebo killed.[/red]")

        # ── Data file actions ──────────────────────────
        elif bid == "view-raw":
            self.populate_file_list("raw")
        elif bid == "view-processed":
            self.populate_file_list("processed")
        elif bid == "process-selected":
            self.process_selected_file()
        elif bid == "delete-selected":
            self.delete_selected_file()

        # ── Global ML pipeline ─────────────────────────
        elif bid == "ml-process":
            self.run_process_data()
        elif bid == "ml-train":
            self.append_log("[yellow]Training global model...[/yellow]")
            self.run_pipeline_async("train")
        elif bid == "ml-full":
            self.append_log("[yellow]Running full pipeline...[/yellow]")
            self.run_pipeline_async("full")

        # ── Terrain ML pipeline ────────────────────────
        elif bid == "ml-train-terrain":
            self.append_log("[yellow]Training terrain models...[/yellow]")
            self.run_pipeline_async("train_terrain")
        elif bid == "ml-full-terrain":
            self.append_log("[yellow]Running full terrain pipeline...[/yellow]")
            self.run_pipeline_async("full_terrain")

        # ── GPS Spoof attacks ──────────────────────────
        elif bid in ("spoof-drift", "spoof-ramp", "spoof-jump", "spoof-static", "spoof-teleport", "spoof-noise"):
            mode_map = {
                "spoof-drift":    "drift",
                "spoof-ramp":     "ramp",
                "spoof-jump":     "jump_drift",
                "spoof-static":   "static",
                "spoof-teleport": "teleport",
                "spoof-noise":    "noise",
            }
            mode = mode_map[bid]
            self.run_spoof_async(mode)

    # ── Async workers ─────────────────────────────────────────────


    @work(exclusive=False)
    async def run_spoof_async(self, mode: str):
        """Run a GPS spoofing attack and stream output to the local spoof-log."""
        cfg = SPOOF_ATTACKS.get(mode)
        if not cfg:
            return
        
        name = cfg["name"]
        cmd  = cfg["command"]
        
        log = self.query_one("#spoof-log", RichLog)
        log.write(f"[bold red]⚡ Launching {name}...[/bold red]")
        
        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(PROJECT_ROOT),
        )
        
        async for line in proc.stdout:
            text = line.decode(errors="replace").rstrip()
            if text:
                # Basic coloring for the spoofer output
                if "[!]" in text:
                    log.write(f"[bold yellow]{text}[/bold yellow]")
                elif "[+]" in text:
                    log.write(f"[green]{text}[/green]")
                elif "DRIFT" in text or "RAMP" in text or "JUMP" in text:
                    log.write(f"[cyan]{text}[/cyan]")
                else:
                    log.write(f"[dim]{text}[/dim]")
        
        await proc.wait()
        log.write(f"[bold green]✓ {name} Finished.[/bold green]")
        log.write("-" * 40)

    @work(exclusive=True)
    async def run_process_data(self):
        """Process all raw GPS CSVs with live progress in the log panel."""
        raw_dir   = Path(str(RAW_DATA_DIR))
        raw_files = sorted(raw_dir.glob("*.csv"))
        if not raw_files:
            self.append_log("[red]No CSV files found to process.[/red]")
            return
        total      = len(raw_files)
        venv_python = PROJECT_ROOT / "venv" / "bin" / "python"
        self.append_log(f"[cyan]=== Processing {total} file(s) ===[/cyan]")
        for i, raw_file in enumerate(raw_files):
            pct = int((i + 1) / total * 100)
            bar = "█" * (pct // 2) + "░" * (50 - pct // 2)
            self.append_log(f"[cyan][{bar}] {pct}%[/cyan]  {raw_file.name}")
            proc = await asyncio.create_subprocess_exec(
                str(venv_python), "-m", "Step_4_Detection.pipeline",
                "process-single", str(raw_file),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(PROJECT_ROOT),
            )
            _, stderr = await proc.communicate()
            if proc.returncode == 0:
                self.append_log(f"[green]✓[/green] {raw_file.name}")
            else:
                err = stderr.decode()[:120] if stderr else ""
                self.append_log(f"[red]✗[/red] {raw_file.name}: {err}")
        self.append_log("[green]=== Processing complete! ===[/green]")
        self.populate_file_list("processed")

    @work(exclusive=False)
    async def run_pipeline_async(self, pipeline_id: str):
        """Run any ML_PIPELINE entry asynchronously, streaming output to log."""
        cfg  = ML_PIPELINE.get(pipeline_id)
        if cfg is None:
            self.append_log(f"[red]Unknown pipeline id: {pipeline_id}[/red]")
            return
        name = cfg["name"]
        cmd  = cfg["command"]
        self.append_log(f"[cyan]>>> {name}[/cyan]")
        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(PROJECT_ROOT),
        )
        async for line in proc.stdout:
            text = line.decode(errors="replace").rstrip()
            if text:
                self.append_log(text)
        await proc.wait()
        if proc.returncode == 0:
            self.append_log(f"[green]=== {name} complete! ===[/green]")
            # Refresh terrain table after any terrain pipeline finishes
            if "terrain" in pipeline_id:
                self.populate_terrain_table()
                self.populate_model_list()
        else:
            self.append_log(f"[red]=== {name} FAILED (exit {proc.returncode}) ===[/red]")

    # ── File actions ──────────────────────────────────────────────

    def process_selected_file(self):
        lv = self.query_one("#file-list", ListView)
        if lv.index is None:
            self.append_log("[red]No file selected.[/red]")
            return
        raw_files = sorted(Path(str(RAW_DATA_DIR)).glob("*.csv"))
        if lv.index >= len(raw_files):
            self.append_log("[red]Selection out of range.[/red]")
            return
        selected = raw_files[lv.index]
        self.append_log(f"[cyan]Processing:[/cyan] {selected.name}")
        result = subprocess.run(
            [str(PROJECT_ROOT / "venv" / "bin" / "python"),
             "-m", "Step_4_Detection.pipeline", "process-single", str(selected)],
            cwd=str(PROJECT_ROOT), capture_output=True, text=True,
        )
        if result.returncode == 0:
            self.append_log(f"[green]✓ Done:[/green] {selected.name}")
        else:
            err = (result.stderr or result.stdout)[:120]
            self.append_log(f"[red]✗ Failed:[/red] {selected.name}: {err}")

    def delete_selected_file(self):
        lv = self.query_one("#file-list", ListView)
        if lv.index is None:
            self.append_log("[red]No file selected.[/red]")
            return
        raw_files = sorted(Path(str(RAW_DATA_DIR)).glob("*.csv"))
        if lv.index >= len(raw_files):
            self.append_log("[red]Selection out of range.[/red]")
            return
        selected = raw_files[lv.index]
        selected.unlink()
        self.append_log(f"[red]Deleted:[/red] {selected.name}")
        self.populate_file_list("raw")

    # ── Component start/stop ──────────────────────────────────────

    def start_selected(self):
        lv = self.query_one("#component-list", ListView)
        if lv.index is not None:
            comp_id = list(COMPONENTS.keys())[lv.index]
            ok, msg = start_component(comp_id)
            self.append_log(msg)
            self.refresh_table()

    def stop_selected(self):
        lv = self.query_one("#component-list", ListView)
        if lv.index is not None:
            comp_id = list(COMPONENTS.keys())[lv.index]
            ok, msg = stop_component(comp_id)
            self.append_log(msg)
            self.refresh_table()

    # ── Key bindings ──────────────────────────────────────────────

    BINDINGS = [
        Binding("q", "quit",    "Quit"),
        Binding("r", "refresh", "Refresh"),
    ]

    def action_refresh(self):
        self.refresh_table()
        self.show_data_info()
        self.populate_terrain_table()
        self.append_log("[cyan]Refreshed.[/cyan]")


def run_app():
    app = GPSDetectorApp()
    app.run()
