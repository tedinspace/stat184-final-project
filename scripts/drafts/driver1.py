from skyfield.api import load
import matplotlib.pyplot as plt

from SSN_RL.environment.ScenarioConfigs import ScenarioConfigs
from SSN_RL.environment.Sensor import SensorResponse
from SSN_RL.environment.Satellite import Satellite, Maneuver
from SSN_RL.agent.Agent import AgentWrapper
from SSN_RL.environment.StateCatalog import StateCatalog
from SSN_RL.scenarioBuilder.clusters import MUOS_CLUSTER
from SSN_RL.scenarioBuilder.SSN import MHR
from SSN_RL.debug.Loggers import EventCounter
from SSN_RL.utils.time import hrsAfterEpoch

EC = EventCounter()

# init configs (epoch + length)
sConfigs = ScenarioConfigs(load.timescale().utc(2024, 11, 24, 0, 0, 0), 12)

# generate true satellites
S = {}
allSatNames = []
for i in [0, 2, 4]: # visibile by MHR
    muos_i = MUOS_CLUSTER[i]
    tmp =  Satellite(muos_i[0], muos_i[1], muos_i[2], sConfigs )
    tmp.reEpoch_init(60*i) # re-epoch 
    S[tmp.name] = tmp
    allSatNames.append(tmp.name)
# truth maneuver
S[allSatNames[0]].addManeuvers([Maneuver(10, 1.5, sConfigs), Maneuver(15.3, 4.15, sConfigs)])
S[allSatNames[2]].addManeuvers([Maneuver(5.5,8.5, sConfigs)])

# init supporting objects ... 
# - sensors
sensors = {MHR.name: MHR}
# - agents 
A = [AgentWrapper("agent 1", allSatNames, [MHR.name])]

# - catalog 
C = StateCatalog(S) # we are initializing with the truth states; this doesn't have to be the case

# scenario loop 

cTime = sConfigs.scenarioEpoch


uniqueManeuverDetections_states = []

while cTime < sConfigs.scenarioEnd:
    # 1. everything needs to move forward to the current time 
    # --> compute truth (and any maneuvers if necessary)
    for satKey in S:
        S[satKey].tick(cTime)
        
    # 2. gather information
    # i. check to see if any sensor events are available to agent 
    events = []
    for sensor in sensors:
        events += sensors[sensor].checkForUpdates(cTime)

    # ii. conduct catalog state updates 
    curratedEvents = [] # events we want to send to agent as a state
    for event in events:
        EC.increment(event.type)
        #print(event.satID + "-->"+ str(event.type))
        if event.type == SensorResponse.COMPLETED_NOMINAL:
            # --> update catalog 
            C.updateState(event.satID, event.newState)
            curratedEvents.append(event)
        elif event.type == SensorResponse.COMPLETED_MANEUVER:
            # --> update catalog with maneuver
            if not C.wasManeuverAlreadyDetected(cTime, event.satID, event.newState): 
                EC.increment("unique maneuver detection")
                uniqueManeuverDetections_states.append(event)  
            else:
                # if not unique, just credit as state update
                event.type = SensorResponse.COMPLETED_NOMINAL
            curratedEvents.append(event)

            C.updateState(event.satID, event.newState)
        elif event.type == SensorResponse.DROPPED_LOST:
            # --> lost
            if not C.wasManeuverAlreadyDetected(cTime, event.satID, event.crystalBallState): 
                curratedEvents.append(event)
            else:
                print("object saved just in time")
        
        else: 
            curratedEvents.append(event)
            
   

    # 3. get agent's responses
    actions = {}
    for agent in A:
        actions[agent.agentID]=agent.decide(cTime, curratedEvents, C)
    
    # 4. execute actions 
    # i. send new tasks to appropriate sensors (delays)
    for agentKey in actions:
        for sat in actions[agentKey]:
            if actions[agentKey][sat]: # if not do nothing (i.e. False)
                sensors[actions[agentKey][sat]].sendSensorTask(cTime, agentKey, sat, C.currentCatalog[sat])
                EC.increment("tasks sent")
    # ii. execute or continue executing tasks that have arrived to sensor
    for sensor in sensors:
        sensors[sensor].tick(cTime, S)
    
    # iterate
    cTime += sConfigs.timeDelta




# - - - - - - - - - - - - - - - - SCENARIO VISUALIZATION  - - - - - - - - - - - - - - - -
EC.display()


def create_enhanced_visualization(sensors, satellites, unique_maneuver_detections, configs):
    """Create an enhanced multi-panel visualization of satellite tracking and maneuvers."""
    
    # Set style and create figure with subplots
    plt.style.use('ggplot')
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(3, 1, height_ratios=[2, 1, 1], hspace=0.3)
    
    # Color scheme with better accessibility
    colors = {
        'MUOS1': '#1f77b4',  # blue
        'MUOS3': '#ff7f0e',  # orange
        'MUOS5': '#9467bd',  # purple
    }
    
    # 1. Main Timeline Plot (top panel)
    ax1 = fig.add_subplot(gs[0])
    
    # Plot sensor tasks with alpha for better overlap visibility
    for task in sensors[MHR.name].completedTasks:
        ax1.plot([task.startTime.tt, task.stopTime.tt], [1, 1], 
                 color=colors[task.satID], linewidth=3, alpha=0.6,
                 label=f'{task.satID} Task')
    
    # Plot actual and detected maneuvers
    for sat_key in satellites:
        # Actual maneuvers
        for m in satellites[sat_key].maneuverList:
            ax1.axvline(x=m.time.tt, color=colors[sat_key], linestyle=':',
                       linewidth=2, label=f'{sat_key} True Maneuver')
    
    # Detected maneuvers
    for event in unique_maneuver_detections:
        ax1.axvline(x=event.arrivalTime.tt, color=colors[event.satID],
                   linestyle='-.', linewidth=2, 
                   label=f'{event.satID} Detected Maneuver')
    
    # Remove duplicate labels
    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys(), 
              bbox_to_anchor=(1.05, 1), loc='upper left')
    
    ax1.set_title('Satellite Tracking Timeline', pad=20, fontsize=12)
    ax1.set_xlabel('Time (hours since epoch)')
    ax1.set_ylabel('Sensor Activity')
    
    # 2. Detection Delay Plot (middle panel)
    ax2 = fig.add_subplot(gs[1])
    plot_detection_delays(ax2, satellites, unique_maneuver_detections, colors)
    
    # 3. Coverage Statistics (bottom panel)
    ax3 = fig.add_subplot(gs[2])
    plot_coverage_statistics(ax3, sensors, satellites, configs, colors)
    
    # Adjust layout and display
    plt.tight_layout()
    return fig

def plot_detection_delays(ax, satellites, detections, colors):
    """Plot maneuver detection delays."""
    delays = []
    sat_names = []
    
    for sat_key in satellites:
        for true_man in satellites[sat_key].maneuverList:
            # Find matching detection
            matching_detection = next(
                (d for d in detections if d.satID == sat_key 
                 and abs(d.arrivalTime.tt - true_man.time.tt) < 1.0),
                None
            )
            if matching_detection:
                delay = matching_detection.arrivalTime.tt - true_man.time.tt
                delays.append(delay)
                sat_names.append(sat_key)
    
    # Create bar plot
    bars = ax.bar(sat_names, delays)
    for idx, bar in enumerate(bars):
        bar.set_color(colors[sat_names[idx]])
    
    ax.set_title('Maneuver Detection Delays')
    ax.set_ylabel('Detection Delay (hours)')
    ax.grid(True, alpha=0.3)

def plot_coverage_statistics(ax, sensors, satellites, configs, colors):
    """Plot sensor coverage statistics."""
    coverage_data = {}
    
    # Calculate coverage percentage for each satellite
    total_time = (configs.scenarioEnd.tt - configs.scenarioEpoch.tt)
    
    for sat_key in satellites:
        observed_time = sum(
            task.stopTime.tt - task.startTime.tt
            for task in sensors[MHR.name].completedTasks
            if task.satID == sat_key
        )
        coverage_data[sat_key] = (observed_time / total_time) * 100
    
    # Create bar plot
    bars = ax.bar(coverage_data.keys(), coverage_data.values())
    for idx, bar in enumerate(bars):
        bar.set_color(colors[list(coverage_data.keys())[idx]])
    
    ax.set_title('Satellite Coverage Statistics')
    ax.set_ylabel('Coverage (%)')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)

# Usage in main code
fig = create_enhanced_visualization(sensors, S, uniqueManeuverDetections_states, sConfigs)
plt.savefig('satellite_tracking_analysis.png', dpi=300, bbox_inches='tight')
plt.show()
