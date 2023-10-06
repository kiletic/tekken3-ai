local network_parser = require('scripts/protocol')
local client = require('scripts/client')
local memory_handler = require('scripts/memory')
local util = require('scripts/util')

local function reload()
  PCSX.pauseEmulator()
  package.loaded["scripts/protocol"] = nil
  package.loaded["scripts/client"] = nil
  package.loaded["scripts/memory"] = nil
  package.loaded["scripts/util"] = nil
  dofile("scripts/main.lua")
end

function DrawImguiFrame()
  local show = imgui.Begin('Tekken3 internals', true)
  if not show then imgui.End() return end

  util.doSliderInt(memory_handler.address.player1_hp, 'Player 1 hp', 0, 130)
  util.doSliderInt(memory_handler.address.player2_hp, 'Player 2 hp', 0, 130)
  util.doSliderInt(memory_handler.address.match_status, 'Match status (0 - not started, 1 - started)', 0, 1)
  util.doSliderInt(memory_handler.address.game_paused, 'Is game paused? (0 - no, 1 - yes)', 0, 1)

  if client then
    client:toggle_status("TCP Client")

    if client.connected and network_parser.observations_to_send > 0 then
      network_parser:send_observation(client)
      network_parser.observations_to_send = network_parser.observations_to_send - 1
    end
  end

  if util.set_keys_cb_countdown > 0 then
    util.set_keys_cb_countdown = util.set_keys_cb_countdown - 1
    if util.set_keys_cb_countdown == 0 then
      util.set_keys_cb()
    end
  end

  if imgui.Button("Reload") then
    reload()
  end

  if imgui.Button("Save state") then
    local save = PCSX.createSaveState()
    local save_state = Support.File.open("haha.slice", "TRUNCATE")
    save_state:writeMoveSlice(save)
    save_state:close()
    print("Saved")
  end

  if imgui.Button("Load state 1") then
    local save_state = Support.File.open("king_vs_law_medium_stage1.slice")
    PCSX.loadSaveState(save_state)
    save_state:close()
    print("Loaded")
  end

  imgui.End()
end
