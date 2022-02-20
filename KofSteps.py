from KofActions import Actions
def set_difficulty(frame_ratio, difficulty):
    steps = [
        {"wait": 0, "actions": [Actions.SERVICE]},
        {"wait": int(10/frame_ratio), "actions": [Actions.P1_DOWN]},
        {"wait": int(10/frame_ratio), "actions": [Actions.P1_DOWN]},
        {"wait": int(10/frame_ratio), "actions": [Actions.P1_DOWN]},
        {"wait": int(10/frame_ratio), "actions": [Actions.P1_DOWN]},
        {"wait": int(10/frame_ratio), "actions": [Actions.P1_DOWN]},
        {"wait": int(10/frame_ratio), "actions": [Actions.P1_DOWN]},
        {"wait": int(10/frame_ratio), "actions": [Actions.P1_A]},
        {"wait": int(10/frame_ratio), "actions": [Actions.P1_DOWN]},
        {"wait": int(10/frame_ratio), "actions": [Actions.P1_A]}]
    if (difficulty % 8) < 3:
        steps += [{"wait": int(10/frame_ratio), "actions": [Actions.P1_LEFT]} for i in range(3-(difficulty % 8))]
    else:
        steps += [{"wait": int(10/frame_ratio), "actions": [Actions.P1_RIGHT]} for i in range((difficulty % 8)-3)]
    steps += [
        {"wait": int(10/frame_ratio), "actions": [Actions.P1_DOWN]},
        {"wait": int(10/frame_ratio), "actions": [Actions.P1_DOWN]},
        {"wait": int(10/frame_ratio), "actions": [Actions.P1_DOWN]},
        {"wait": int(10/frame_ratio), "actions": [Actions.P1_DOWN]},
        {"wait": int(10/frame_ratio), "actions": [Actions.P1_DOWN]},
        {"wait": int(10/frame_ratio), "actions": [Actions.P1_DOWN]},
        {"wait": int(10/frame_ratio), "actions": [Actions.P1_A]},
        {"wait": int(10/frame_ratio), "actions": [Actions.P1_DOWN]},
        {"wait": int(10/frame_ratio), "actions": [Actions.P1_A]},
        {"wait": int(10/frame_ratio), "actions": [Actions.P1_DOWN]},
        {"wait": int(10/frame_ratio), "actions": [Actions.P1_DOWN]},
        {"wait": int(10/frame_ratio), "actions": [Actions.P1_A]},
        {"wait": int(10/frame_ratio), "actions": [Actions.P1_DOWN]},
        {"wait": int(10/frame_ratio), "actions": [Actions.P1_DOWN]},
        {"wait": int(10/frame_ratio), "actions": [Actions.P1_DOWN]},
        {"wait": int(10/frame_ratio), "actions": [Actions.P1_A]}]
    return steps
def start_game(frame_ratio):
    return [
        {"wait": int(300/frame_ratio), "actions": [Actions.COIN_P1]},
        {"wait": int(10/frame_ratio), "actions": [Actions.COIN_P1]},
        {"wait": int(600/frame_ratio), "actions": [Actions.P1_START]},
        {"wait": int(300 / frame_ratio), "actions": [Actions.P1_A]},
        {"wait": int(300 / frame_ratio), "actions": [Actions.P1_A]},
        {"wait": int(300/frame_ratio), "actions": [Actions.P1_A]},
        {"wait": int(300/frame_ratio), "actions": [Actions.P1_A]},
        {"wait": int(300/frame_ratio), "actions": [Actions.P1_A]}]