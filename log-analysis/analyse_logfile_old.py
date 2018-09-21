#!/usr/bin/env python
import csv
import numpy as np

if __name__ == '__main__':
    folder = "log-analysis/"
    log = open("noGazebo_ddpg.log", "r")
    rolloutFile = open(folder + "rollout-steptimes.csv", "w")
    rolloutTimes = csv.writer(rolloutFile) #, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    fittingFile = open(folder + "fittingtimes.csv", "w")
    fittingTimes = csv.writer(fittingFile)
    savingFile = open(folder + "savingtimes.csv", "w")
    savingTimes = csv.writer(savingFile)
    cycleFile = open(folder + "cycletimes.csv", "w")
    cycleTimes = csv.writer(cycleFile)
    epochFile = open(folder + "epochtimes.csv", "w")
    epochTimes = csv.writer(epochFile)
    actionFile = open(folder + "actions.csv", "w")
    actions = csv.writer(actionFile)
    rewardFile = open(folder + "rewards.csv", "w")
    rewards = csv.writer(rewardFile)
    i = 0
    graspCount = 0
    for line in log:
        text = str(line)
        # print(text)
        if "runtime rollout" in text:
            # print(i)
            i += 1
            rolloutTime = text.split("step ")[1]
            timeWithID = rolloutTime.split(": ")
            stepID = timeWithID[0]
            time = float(timeWithID[1].split("s")[0])
            rolloutTimes.writerow([stepID, time])
        elif "runtime training" in text:
            fittingTime = text.split("critic: ")[1]
            fittingTime = float(fittingTime.split("s")[0])
            fittingTimes.writerow([fittingTime])
        elif "runtime saving" in text:
            savingTime = text.split("saving: ")[1]
            savingTime = float(savingTime.split("s")[0])
            savingTimes.writerow([savingTime])
        elif "runtime epoch-cycle" in text:
            cycleTimeWithID = text.split("cycle ")[1]
            cycleTimeWithID = cycleTimeWithID.split(": ")
            cycleID = int(cycleTimeWithID[0])
            cycleTime = float(cycleTimeWithID[1].split("s")[0])
            cycleTimes.writerow([cycleID, cycleTime])
        elif "runtime epoch" in text:
            epochTimeWithID = text.split("epoch ")[1]
            epochTimeWithID = epochTimeWithID.split(": ")
            epochID = int(epochTimeWithID[0])
            epochTime = float(epochTimeWithID[1].split("s")[0])
            epochTimes.writerow([epochID, epochTime])
        elif "moving arm to position" in text:
            action = text.split("position: [")[1]
            action = action.split("]")[0]
            action = np.array(action.split()).astype(np.float)
            actions.writerow(action)
        elif "received reward" in text:
            reward = float(text.split("reward: ")[1].split("\n")[0])
            rewards.writerow([reward])
        elif "center of gravity" in text:
            cogCorrect = text.split("fingers: ")[1].split("\n")[0] == "True"
        elif "left gripper crashes" in text:
            leftCrash = text.split("crashes: ")[1].split("\n")[0] == "True"
        elif "right gripper crashes" in text:
            rightCrash = text.split("crashes: ")[1].split("\n")[0] == "True"
        elif "grasping would be successful" in text:
            graspCount += 1



    print("total successful grasps: {}".format(graspCount))
    log.close()
    rolloutFile.close()
    fittingFile.close()
    savingFile.close()
    cycleFile.close()
    epochFile.close()
    actionFile.close()
    rewardFile.close()
    print("terminated")

# if __main___