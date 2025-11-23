from typing import Dict, Tuple, List
from parseStats import process_file
from pathlib import Path
import matplotlib.pyplot as plt


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

    print("Generating plot data...")
    plot_data = []

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
                    # workload_data[wl] is (cost, ipc)
                    ipcs_targeted.append(workload_data[wl][1])

        # If no targeted workloads found (or list empty), mean is 0
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
        print("Warning: No plot data found! (Check if files were parsed correctly)")
        return

    # Sort data by Cost (ascending)
    plot_data.sort(key=lambda x: x["cost"])

    # --- Generate Configuration Names and Write Text File ---

    config_file_lines = []

    for idx, data_point in enumerate(plot_data):
        config_num = idx + 1
        cost = data_point["cost"]
        new_label = f"Configuration {config_num} (Cost {cost})"

        # Update the label in the plot data
        data_point["label"] = new_label

        # Retrieve the detailed dictionary
        original_key = data_point["original_params"]
        param_details = paramsByConfig.get(original_key, {})

        # Format for text file
        config_file_lines.append(f"{new_label}:")
        for key in sorted(param_details.keys()):
            config_file_lines.append(f"    {key}: {param_details[key]}")
        config_file_lines.append("")  # Empty line for spacing

    # Save Configurations.txt
    with open(f"{plotOut}-Configurations.txt", "w") as f:
        f.write("\n".join(config_file_lines))
    print("Saved configuration details to Configurations.txt")

    # --- Plotting ---

    labels = [d["label"] for d in plot_data]

    # Extract datasets
    ipc_all = [d["ipc_all"] for d in plot_data]
    ipc_targeted = [d["ipc_targeted"] for d in plot_data]

    eff_all = [d["efficiency_all"] for d in plot_data]
    eff_targeted = [d["efficiency_targeted"] for d in plot_data]

    # Bar Chart Settings
    bar_height = 0.35
    y_indices = range(len(labels))
    # Offset bars: "All" goes slightly up (-), "Targeted" goes slightly down (+)
    y_pos_all = [y - bar_height/2 for y in y_indices]
    y_pos_targeted = [y + bar_height/2 for y in y_indices]

    plot_path = Path(plotOut)
    stem = plot_path.stem
    suffix = plot_path.suffix
    parent = plot_path.parent

    # 1. Plot IPC
    plt.figure(figsize=(12, 10))  # Wider figure for legend

    plt.barh(y_pos_all, ipc_all, height=bar_height,
             label='All Workloads', color='skyblue', edgecolor='black')
    plt.barh(y_pos_targeted, ipc_targeted, height=bar_height,
             label='Targeted Workloads', color='orange', edgecolor='black')

    plt.yticks(y_indices, labels)
    plt.xlabel('Geometric Mean IPC')
    plt.title('Performance (IPC) by Configuration')
    plt.legend()
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(plotOut)
    print(f"Saved IPC plot to {plotOut}")
    plt.close()

    # 2. Plot Efficiency (IPC / Cost)
    cost_plot_filename = parent / f"{stem}_perCost{suffix}"

    plt.figure(figsize=(12, 10))

    plt.barh(y_pos_all, eff_all, height=bar_height,
             label='All Workloads', color='lightgreen', edgecolor='black')
    plt.barh(y_pos_targeted, eff_targeted, height=bar_height,
             label='Targeted Workloads', color='gold', edgecolor='black')

    plt.yticks(y_indices, labels)
    plt.xlabel('Efficiency (IPC / Cost)')
    plt.title('Cost Efficiency by Configuration')
    plt.legend()
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(cost_plot_filename)
    print(f"Saved Efficiency plot to {cost_plot_filename}")
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
