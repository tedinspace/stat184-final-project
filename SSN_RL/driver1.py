from skyfield.api import load
from environment.ScenarioConfigs import ScenarioConfigs
from environment.Satellite import Satellite
from environment.Manuever import Maneuver
from environment.Sensor import Sensor
from scenarioBuilder.clusters import MUOS_CLUSTER

# init configs (epoch + length)
sConfigs = ScenarioConfigs(load.timescale().utc(2024, 11, 24, 0, 0, 0), 10.5)

# generate true satellites
nSats = len(MUOS_CLUSTER)
S = []
for i in range(nSats):
    muos_i = MUOS_CLUSTER[i]
    tmp =  Satellite(muos_i[0], muos_i[1], muos_i[2], sConfigs )
    tmp.reEpoch_init(60*i) # re-epoch 
    S.append(tmp)
# truth maneuver
S[1].addManeuvers([Maneuver(10, 1.5, sConfigs), Maneuver(15.3, 4.15, sConfigs)])
S[2].addManeuvers([Maneuver(5.5,8.5, sConfigs)])

# init supporting objects ... 
# - sensors
MHR = Sensor("mhr", [42.3583, 71.060, 100])
ASCENSION = Sensor("ascension", [-7.678805483927795, -13.265374101627982, 0])
MAUI = Sensor("GEODSS Maui", [20.708259458876853, -156.2567944944128, 0])
# - agents 
# - catalog 
was_seen = set()
# scenario loop 
cTime = sConfigs.scenarioEpoch
while cTime < sConfigs.scenarioEnd:
    # 1. everything needs to move forward to the current time 
    # --> compute truth (and any maneuvers if necessary)
    S_truth = []
    for s in S:
        X = s.tick(cTime)
        S_truth.append(X)
        if MHR.isVisible(X, cTime) or ASCENSION.isVisible(X, cTime) or MAUI.isVisible(X, cTime):
            was_seen.add(s.name)


    # 2. translate knowable states to agents
    
    # 3. get agent's responses

    # 4. stage responses impact/rewards/punishments
    
    # iterate
    cTime += sConfigs.timeDelta

print(was_seen)
