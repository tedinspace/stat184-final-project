from skyfield.api import load
from environment.ScenarioConfigs import ScenarioConfigs
from environment.Satellite import Satellite
from environment.Manuever import Maneuver
from scenarioBuilder.clusters import MUOS_CLUSTER


sConfigs = ScenarioConfigs(load.timescale().utc(2024, 11, 24, 0, 0, 0), 24.5)

muos1 = MUOS_CLUSTER[0]

S1 = Satellite(muos1[0], muos1[1], muos1[2], sConfigs )
S1.reEpoch_init(60)
S1.addManeuvers([Maneuver(10, 1.5, sConfigs), Maneuver(15.3, 4.15, sConfigs)])

cTime = sConfigs.scenarioEpoch
while cTime < sConfigs.scenarioEnd:
    # 1. everything needs to move forward to the current time 
    X1 = S1.tick(cTime)

    X1_no = S1.noEventObject_debugging.at(cTime)
    sb_no = X1_no.subpoint()
    sb = X1.subpoint()
    # 2. translate knowable states to agents
    
    # 3. get agent's responses

    # 4. stage responses impact/rewards/punishments
    
    # iterate
    cTime += sConfigs.timeDelta


