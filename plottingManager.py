from typing import Dict, Tuple, List
from parseStats import process_file
from pathlib import Path
import matplotlib.pyplot as plt

defaultCpuParameters = {
    "--fetch_buffer_size": "64",
    "--fetch_queue_size": "4",
    "--fetch_width": "1",
    "--decode_width": "1",
    "--rename_width": "1",
    "--dispatch_width": "1",
    "--issue_width": "1",
    "--commit_width": "1",
    "--num_iq_entries": "16",
    "--num_rob_entries": "32",
    "--lq_entries": "4",
    "--sq_entries": "4",
    "--fu_pool": "basic"
}


BENCHMARKS = [
    "basicmath",
    "bitcounts",
    "qsort",
    'susan_edges',
    'susan_smoothing',
    'susan_corners',
    'jpeg_encode',
    'jpeg_decode',
    "dijkstra",
]

# Categorization
BRANCH_SIMS = ["qsort", "dijkstra"]
ALU_SIMS = ["bitcounts"]
MEMORY_SIMS = ['susan_edges',
               'susan_smoothing',
               'susan_corners',
               'jpeg_encode',
               'jpeg_decode']

BRANCH_DIR = "./branchResults"
ALU_DIR = "./ALUResults/"
MEMORY_DIR = "./memoryResults/"

AVAILABLE_PARAMS = [
    "--fetch_buffer_size",
    "--fetch_queue_size",
    "--fetch_width",
    "--decode_width",
    "--rename_width",
    "--dispatch_width",
    "--issue_width",
    "--commit_width",
    "--num_iq_entries",
    "--num_rob_entries",
    "--lq_entries",
    "--sq_entries",
    "--fu_pool"
]

AVAILABLE_PARAMS.sort()

COST_LOOKUP = {
    "--fetch_buffer_size": {
        "64": 20,  "64 B": 20,
        "128": 50, "128 B": 50
    },
    "--fetch_queue_size": {
        "4": 20,
        "8": 50,
        "16": 100,
        "32": 140,
        "64": 240,
    },
    "--fetch_width": {
        "1": 15, "2": 50, "4": 150
    },
    "--decode_width": {
        "1": 15, "2": 50, "4": 150
    },
    "--rename_width": {
        "1": 15, "2": 50, "4": 150
    },
    "--dispatch_width": {
        "1": 15, "2": 50, "4": 150
    },
    "--issue_width": {
        "1": 15, "2": 50, "4": 150
    },
    "--commit_width": {
        "1": 15, "2": 50, "4": 150
    },
    "--num_iq_entries": {
        "16": 40,
        "32": 80,
        "64": 180
    },
    "--num_rob_entries": {
        "32": 50,
        "64": 120,
        "128": 260
    },
    "--lq_entries": {
        "4": 10,
        "8": 30,
        "16": 70,
        "32": 140
    },
    "--sq_entries": {
        "4": 10,
        "8": 30,
        "16": 70,
        "32": 140
    },
    "--fu_pool": {
        "basic": 80,
        "extended": 140,
        "aggressive": 320
    }
}


def generateDiffLabel(config: Dict[str, str]) -> str:
    """
    Generates a label string containing only parameters that differ 
    from the defaultCpuParameters in TestGroup.py.
    """
    changes = []

    # Sort keys for consistent label ordering
    for key in sorted(config.keys()):
        val = config[key]
        default_val = defaultCpuParameters.get(key)

        # Compare strings (filename parsing yields strings, TestGroup defaults are strings)
        if default_val is None or val != default_val:
            # Strip '--' for cleaner plot labels
            clean_name = key.lstrip('-')
            changes.append(f"{clean_name}={val}")

    if not changes:
        return "Default Config"

    return ", ".join(changes)


def calculateParameterCost(cpuConfig: Dict[str, str]) -> int:
    total_cost = 0

    for param_name, param_value in cpuConfig.items():
        if param_name not in COST_LOOKUP:
            raise ValueError(f"Unknown parameter: '{param_name}'")

        value_options = COST_LOOKUP[param_name]

        if param_value in value_options:
            cost = value_options[param_value]
            total_cost += cost
        else:
            raise ValueError(
                f"Invalid value '{param_value}' for parameter '{param_name}'. "
                f"Allowed values: {list(value_options.keys())}"
            )

    return total_cost


def parseFileName(filename: str) -> Tuple[str, str, str]:
    paramValues = filename.rstrip(".txt").split("-")
    if (len(paramValues) != len(AVAILABLE_PARAMS) + 1):
        raise ValueError("not all params included in filename")
    simName = paramValues[0]
    paramDict = {}
    for idx, paramName in enumerate(AVAILABLE_PARAMS):
        paramDict[paramName] = paramValues[idx + 1]
    paramString = "-".join(paramValues[1:-1])

    return simName, paramDict, paramString


def parseIPC(file: Path) -> float:
    path = Path(file)
    result = process_file(path, 1)
    if result is None:
        raise ValueError(f"Unable to parse {file.name}")
    else:
        return result["ipc"]


# For each group.
# - Find all files in the group
# - For each make two plots IPC and IPC / Performance (pass cost and the results)
#   - They should have one plot for aggregate benchmark and one plot for the
#        group of workloads (geom mean)


def valuesByConfigAndWorkload(fileNames: List[Path]) -> Tuple[Dict[str, Dict[str, Tuple[int, float]]], Dict[str, Dict[str, str]]]:
    simsByConfig = {}
    paramsByConfig = {}  # Store the raw param dicts here

    for file in fileNames:
        workload, paramDict, paramString = parseFileName(file.name)
        if paramString not in simsByConfig:
            simsByConfig[paramString] = {}
            paramsByConfig[paramString] = paramDict  # Save paramDict once

        # Calculate our paramCost and IPC from the file
        simsByConfig[paramString][workload] = (
            calculateParameterCost(paramDict),
            parseIPC(file)
        )

    # Assert we have all workloads for each config
    for config in simsByConfig:
        if set(simsByConfig[config].keys()) != set(BENCHMARKS):
            missing = set(BENCHMARKS) - set(simsByConfig[config].keys())
            raise ValueError(f"Missing Simulations for config: {
                             config}\nMissing Workloads: {missing}")

    return simsByConfig, paramsByConfig


def getFiles(dir: Path) -> List[Path]:
    if not dir.exists():
        raise FileNotFoundError(f"The path '{dir}' does not exist.")
    if not dir.is_dir():
        raise NotADirectoryError(f"The path '{dir}' is not a directory.")

    # Iterate through the directory, filter for files, and check for "-" in the name
    return [
        f for f in dir.iterdir()
        if f.is_file() and "-" in f.name
    ]


def geomMean(values: List[float]) -> float:
    product = 1
    for value in values:
        product *= value
    return (pow(product, 1 / len(values)))


def makePlotsForTestGroup(targetedWorkloads: List[str],
                          ipcAndCost: Dict[str, Dict[str, Tuple[int, float]]],
                          paramsByConfig: Dict[str, Dict[str, str]],
                          plotOut: str):

    print(f"Generating combined plot data for {plotOut}...")
    plot_data = []

    # --- Data Processing ---
    for paramsString, workload_data in ipcAndCost.items():
        if not workload_data:
            continue

        # 1. Calculate Geometric Mean for ALL Workloads
        current_entries = list(workload_data.values())
        ipcs_all = [ipc for (cost, ipc) in current_entries]
        config_cost = current_entries[0][0]
        gmean_ipc_all = geomMean(ipcs_all)

        # 2. Calculate Geometric Mean for TARGETED Workloads
        ipcs_targeted = []
        if targetedWorkloads:
            for wl in targetedWorkloads:
                if wl in workload_data:
                    ipcs_targeted.append(workload_data[wl][1])

        gmean_ipc_targeted = geomMean(ipcs_targeted) if ipcs_targeted else 0.0

        plot_data.append({
            "original_params": paramsString,
            "cost": config_cost,
            "ipc_all": gmean_ipc_all,
            "ipc_targeted": gmean_ipc_targeted,
            "efficiency_all": gmean_ipc_all / config_cost if config_cost > 0 else 0,
            "efficiency_targeted": gmean_ipc_targeted / config_cost if config_cost > 0 else 0
        })

    if not plot_data:
        print("Warning: No plot data found!")
        return

    # Sort data by Cost (ascending)
    plot_data.sort(key=lambda x: x["cost"])

    # --- Prepare Table Data ---
    all_config_dicts = [paramsByConfig[d["original_params"]]
                        for d in plot_data]

    # Identify varying parameters
    varying_keys = []
    for key in AVAILABLE_PARAMS:
        default_val = defaultCpuParameters.get(key)
        is_varying = any(
            cfg.get(key) != default_val for cfg in all_config_dicts)
        if is_varying:
            varying_keys.append(key)

    # Prepare lists for the table
    table_cell_text = []
    table_row_labels = []
    table_col_labels = ["Cost"] + [k.lstrip('-') for k in varying_keys]

    config_file_lines = []

    for idx, data_point in enumerate(plot_data):
        config_name = f"Config {idx + 1}"
        data_point["label"] = config_name

        # Text file details
        original_key = data_point["original_params"]
        param_details = paramsByConfig.get(original_key, {})

        config_file_lines.append(f"{config_name} (Cost {data_point['cost']}):")
        for key in sorted(param_details.keys()):
            config_file_lines.append(f"    {key}: {param_details[key]}")
        config_file_lines.append("")

        # Table Row
        row_values = [str(data_point["cost"])]
        for k in varying_keys:
            row_values.append(param_details.get(k, "-"))

        table_row_labels.append(config_name)
        table_cell_text.append(row_values)

    # Save Configurations.txt
    with open(f"{plotOut}-Configurations.txt", "w") as f:
        f.write("\n".join(config_file_lines))

    # --- Plotting with GridSpec (Top: Plots, Bottom: Table) ---

    labels = [d["label"] for d in plot_data]
    ipc_all = [d["ipc_all"] for d in plot_data]
    ipc_targeted = [d["ipc_targeted"] for d in plot_data]
    eff_all = [d["efficiency_all"] for d in plot_data]
    eff_targeted = [d["efficiency_targeted"] for d in plot_data]

    y_indices = range(len(labels))
    bar_height = 0.35
    y_pos_all = [y - bar_height/2 for y in y_indices]
    y_pos_targeted = [y + bar_height/2 for y in y_indices]

    # Initialize Figure
    fig = plt.figure(figsize=(18, 9))

    # Create Grid: 2 Rows, 2 Cols
    # height_ratios=[4, 1]: Top row (plots) gets ~80% height, Bottom row (table) gets ~20%
    # hspace=0.6: Increased space between the plots and the table
    gs = fig.add_gridspec(2, 2, height_ratios=[4, 1], hspace=0.6, wspace=0.1)

    # --- Top Left: IPC Plot ---
    ax_ipc = fig.add_subplot(gs[0, 0])

    ax_ipc.barh(y_pos_all, ipc_all, height=bar_height,
                label='All Workloads', color='skyblue', edgecolor='black')
    ax_ipc.barh(y_pos_targeted, ipc_targeted, height=bar_height,
                label='Targeted Workloads', color='orange', edgecolor='black')

    ax_ipc.set_yticks(y_indices)
    ax_ipc.set_yticklabels(labels)
    ax_ipc.set_xlabel('Geometric Mean IPC')
    ax_ipc.legend(loc='lower right')
    ax_ipc.grid(axis='x', linestyle='--', alpha=0.7)
    ax_ipc.set_ylim(-0.5, len(labels) - 0.5)

    # --- Top Right: Efficiency Plot ---
    ax_eff = fig.add_subplot(gs[0, 1], sharey=ax_ipc)

    ax_eff.barh(y_pos_all, eff_all, height=bar_height,
                label='All Workloads', color='lightgreen', edgecolor='black')
    ax_eff.barh(y_pos_targeted, eff_targeted, height=bar_height,
                label='Targeted Workloads', color='gold', edgecolor='black')

    ax_eff.tick_params(axis='y', left=False, labelleft=False)
    ax_eff.set_xlabel('Efficiency (IPC / Cost)')
    ax_eff.legend(loc='lower right')
    ax_eff.grid(axis='x', linestyle='--', alpha=0.7)

    # --- Bottom: Table (Spanning Full Width) ---
    ax_table = fig.add_subplot(gs[1, :])
    ax_table.axis('off')

    the_table = ax_table.table(cellText=table_cell_text,
                               rowLabels=table_row_labels,
                               colLabels=table_col_labels,
                               loc='center',
                               cellLoc='center')

    the_table.auto_set_font_size(False)
    the_table.set_fontsize(10)  # Reduced font size
    the_table.scale(1, 1.5)    # Reduced row height scaling

    # --- Save ---
    plt.savefig(plotOut, bbox_inches='tight')
    print(f"Saved combined plot to {plotOut}")
    plt.close()


def main():
    # Ensure this path exists relative to where you run the script!
    target_dir = Path("./memoryResults")
    output_plot = "./plots/memoryPlot.pdf"
    filenames = getFiles(target_dir)
    data, params = valuesByConfigAndWorkload(filenames)
    makePlotsForTestGroup(MEMORY_SIMS, data, params, output_plot)

    target_dir = Path("./ALUResults")
    output_plot = "./plots/ALUPlot.pdf"
    filenames = getFiles(target_dir)
    data, params = valuesByConfigAndWorkload(filenames)
    makePlotsForTestGroup(ALU_SIMS, data, params, output_plot)

    target_dir = Path("./branchResults")
    output_plot = "./plots/branchPlot.pdf"
    filenames = getFiles(target_dir)
    data, params = valuesByConfigAndWorkload(filenames)
    makePlotsForTestGroup(BRANCH_SIMS, data, params, output_plot)


main()
