local memory_handler = require("scripts/memory")

local M = {}

M.keys = {
  SELECT = 0,
  START = 3,
  UP = 4,
  RIGHT = 5,
  DOWN = 6,
  LEFT = 7,
  L2 = 8,
  R2 = 9,
  L1 = 10,
  R1 = 11,
  TRIANGLE = 12,
  CIRCLE = 13,
  CROSS = 14,
  SQUARE = 15,
  NO_ACTION = 16 -- this is fake
}

M.set_keys_cb = nil
M.set_keys_cb_countdown = 0

function M.clear_keys(slot_id)
  for _, v in pairs(M.keys) do
    if v ~= M.keys.NO_ACTION and PCSX.SIO0.slots[slot_id].pads[1].getButton(v) == true then
      PCSX.SIO0.slots[slot_id].pads[1].clearOverride(v)
    end
  end
end

function M.set_key(slot_id, key)
  if key ~= M.keys.NO_ACTION then
    PCSX.SIO0.slots[slot_id].pads[1].setOverride(key)
  end
end


M.states = {
  "king_vs_law_easy_stage1.slice",
  "b.slice",
  "king_vs_law_medium_stage1.slice",
  "king_vs_law_hard_stage1.slice"
}

function M.load_state(state)
  local save_state = Support.File.open(M.states[state])
  PCSX.loadSaveState(save_state)
  save_state:close()
end

function M.doSliderInt(address, name, min, max)
  local value, pointer = memory_handler.read_from_memory(address)
  local changed
  changed, value = imgui.SliderInt(name, value, min, max, '%d')
  if changed then pointer[0] = value end
end

return M
