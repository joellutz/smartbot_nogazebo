#!/usr/bin/env python
import csv
import numpy as np

if __name__ == '__main__':
    folder = "log-analysis/"
    log = open("noGazebo_ddpg.log", "r")
    analysisFile = open(folder + "ddpg-analysis.csv", "w")
    analysis = csv.writer(analysisFile)
    analysis.writerow(["step", "time", "r", "phi", "reward", "boxX", "boxY", "boxPhi", "success"])

    fittingFile = open(folder + "model-fitting-analysis.csv", "w")
    fittings = csv.writer(fittingFile)
    fittings.writerow(["epoch-cycle", "critic-loss", "actor-loss"])

    graspCount = 0
    step = 0
    time = 0.0
    r = 0.0
    phi = 0.0
    reward = 0.0
    boxX = 0.0
    boxY = 0.0
    boxPhi = 0.0
    success = False

    epoch_cycle = 0
    critic_loss = 0.0
    actor_loss = 0.0
    edgeActionCount = 0

    for line in log:
        text = str(line)
        # print(text)
        if "runtime rollout" in text:
            # print(i)
            rolloutTime = text.split("step ")[1]
            timeWithID = rolloutTime.split(": ")
            stepID = timeWithID[0]
            time = float(timeWithID[1].split("s")[0])
            analysis.writerow([step, time, r, phi, reward, boxX, boxY, boxPhi, success])
            step += 1
            success = False
        elif "moving arm to position" in text:
            action = text.split("position: [")[1]
            action = action.split("]")[0]
            action = np.array(action.split()).astype(np.float)
            r = action[0]
            phi = action[1]
        elif "box position" in text:
            boxPos = text.split("position: [")[1]
            boxPos = boxPos.split("]")[0]
            boxPos = np.array(boxPos.split()).astype(np.float)
            boxX = boxPos[0]
            boxY = boxPos[1]
            boxPhi = boxPos[2]
        elif "received reward" in text:
            reward = float(text.split("reward: ")[1].split("\n")[0])
        elif "grasping would be successful" in text:
            graspCount += 1
            success = True
        elif "critic loss" in text:
            critic_loss = float(text.split("critic loss: ")[1].split("\n")[0])
        elif "actor loss" in text:
            actor_loss = float(text.split("actor loss: ")[1].split("\n")[0])
            fittings.writerow([epoch_cycle, critic_loss, actor_loss])
        elif "runtime epoch-cycle" in text:
            epoch_cycle += 1
        elif "selected (unscaled) action" in text:
            action = text.split("action: [")[1]
            action = action.split("]")[0]
            action = np.array(action.split()).astype(np.float)
            a1 = action[0]
            a2 = action[1]
            limit = 0.99
            if(a1 >= limit or a1 <= -limit or a2 >= limit or a2 <= -limit):
                edgeActionCount += 1

    # for

    print("total actions: {}".format(step))
    print("total actions close to action limit: {}".format(edgeActionCount))
    print("total successful grasps: {}".format(graspCount))
    
    log.close()
    analysisFile.close()
    fittingFile.close()
    print("terminated")

# if __main___