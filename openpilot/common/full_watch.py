import time
from cereal import messaging

LOG_PATH = "/data/full_drive.log"

def names(events):
    out = []
    for e in events:
        try:
            out.append(e.name)
        except Exception:
            pass
    return out

services = ['carState', 'controlsState', 'selfdriveState', 'longitudinalPlan', 'radarState', 'carrotMan']
sm = messaging.SubMaster(services)

state = {}

with open(LOG_PATH, "a") as f:
    f.write("\n=== started " + time.strftime('%Y-%m-%d %H:%M:%S') + " ===\n")
    last_write = 0
    while True:
        sm.update(100)

        if sm.updated.get('carState'):
            cs = sm['carState']
            state['v'] = cs.vEgo * 3.6
            state['a'] = cs.aEgo
            state['angle'] = cs.steeringAngleDeg
            state['cs_events'] = names(cs.events)
        if sm.updated.get('longitudinalPlan'):
            lp = sm['longitudinalPlan']
            try:
                state['a_target'] = lp.accels[0] if len(lp.accels) else None
            except Exception:
                pass
        if sm.updated.get('radarState'):
            rs = sm['radarState']
            try:
                lead = rs.leadOne
                state['lead_d'] = lead.dRel if lead.status else None
                state['lead_vrel'] = lead.vRel if lead.status else None
            except Exception:
                pass
        if sm.updated.get('carrotMan'):
            try:
                state['vturn'] = sm['carrotMan'].vTurnSpeed
            except Exception:
                pass

        ev = list(state.get('cs_events', []))
        for svc in ('controlsState', 'selfdriveState'):
            if sm.updated.get(svc):
                try:
                    ev += names(sm[svc].events)
                except Exception:
                    pass

        now = time.time()
        if ev or (now - last_write > 0.5):
            last_write = now
            ts = time.strftime('%Y-%m-%d %H:%M:%S')
            line = (ts + " v=" + format(state.get('v',0), '.1f') +
                    " a=" + format(state.get('a',0), '.2f') +
                    " angle=" + format(state.get('angle',0), '.1f') +
                    " vturn=" + format(state.get('vturn',0), '.1f') +
                    " lead_d=" + str(state.get('lead_d')) +
                    " lead_vrel=" + str(state.get('lead_vrel')) +
                    " a_target=" + str(state.get('a_target')) +
                    " events=" + str(ev) + "\n")
            f.write(line)
            f.flush()
