from SSN_RL.environment.Sensor import SensorResponse

class EventCounter: 
    def __init__(self):
        self.eventCounts = {
            SensorResponse.COMPLETED_NOMINAL: 0, 
            SensorResponse.COMPLETED_MANEUVER: 0, 
            SensorResponse.UNIQUE_MAN: 0,
            SensorResponse.INVALID_TIME: 0,
            SensorResponse.INVALID: 0, 
            SensorResponse.DROPPED_SCHEDULING: 0, 
            SensorResponse.DROPPED_LOST: 0, 
            

        }

    def increment(self, eventType):
        if eventType in self.eventCounts:
            self.eventCounts[eventType]+=1
        else:
            self.eventCounts[eventType]=1


    def display(self):
        print('---'*24)
        for eventKey in self.eventCounts:
            print("> "+str(eventKey)+" - "+str(self.eventCounts[eventKey]))
        print('---'*24)
        
