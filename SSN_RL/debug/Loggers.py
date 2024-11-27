
class EventCounter: 
    def __init__(self):
        self.eventCounts = {}

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
        
