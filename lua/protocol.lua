local memory_handler = require('scripts/memory')
local util = require('scripts/util')

local M = {}

M.observations_to_send = 0

local SendType = {
  OBSERVATION_START = 0,
  SCREENSHOT = 1,
  MATCH_STATUS = 2,
  GAME_PAUSED = 3,
  PLAYER1_HP = 4,
  PLAYER2_HP = 5,
  OBSERVATION_END = 255
}

local ParseType = {
  ACTION = 1,
  RESET = 2,
  OBSERVATIONS_REQUEST = 3,
  CLEAR_CONTROLLER = 4
}

function M:parse(connection, opcode)
  if opcode == ParseType.ACTION then
    self:parse_action(connection)
  elseif opcode == ParseType.RESET then
    self:parse_reset(connection)
  elseif opcode == ParseType.OBSERVATIONS_REQUEST then
    self:parse_observations_request(connection)
  elseif opcode == ParseType.CLEAR_CONTROLLER then
    self:parse_clear_controller(connection)
  else
    print("Invalid opcode: ", opcode)
  end
end

function M:parse_clear_controller(connection)
  local slot_id = connection.client:readU8()
  util.clear_keys(slot_id)
end

function M:parse_reset(connection)
  local state = connection.client:readU8()

  util.clear_keys(1)
  util.clear_keys(2)
  util.load_state(state)
end

function M:parse_action(connection)
  local slot_id = connection.client:readU8()
  local first_key = connection.client:readU8()
  local second_key = connection.client:readU8()

  util.clear_keys(slot_id)
  util.set_keys_cb = function ()
    util.set_key(slot_id, first_key)
    util.set_key(slot_id, second_key)
  end
  util.set_keys_cb_countdown = 2
end

function M:parse_observations_request(connection)
  M.observations_to_send = connection.client:readU8()
end

function M:send_observation(connection)
  if not connection.connected then
    print("Tried to send observation but client is not connected.")
    return
  end

  connection.client:writeU8(SendType.OBSERVATION_START)
  self:send_screenshot(connection)
  self:send_match_status(connection)
  self:send_game_paused(connection)
  self:send_player1_hp(connection)
  self:send_player2_hp(connection)
  connection.client:writeU8(SendType.OBSERVATION_END)
end

function M:send_match_status(connection)
  connection.client:writeU8(SendType.MATCH_STATUS)
  local match_status, _ = memory_handler.read_from_memory(memory_handler.address.match_status)
  connection.client:writeU8(match_status)
end

function M:send_game_paused(connection)
  connection.client:writeU8(SendType.GAME_PAUSED)
  local game_paused, _ = memory_handler.read_from_memory(memory_handler.address.game_paused)
  connection.client:writeU8(game_paused)
end

function M:send_player1_hp(connection)
  connection.client:writeU8(SendType.PLAYER1_HP)
  local player1_hp, _ = memory_handler.read_from_memory(memory_handler.address.player1_hp)
  connection.client:writeU8(player1_hp)
end

function M:send_player2_hp(connection)
  connection.client:writeU8(SendType.PLAYER2_HP)
  local player2_hp, _ = memory_handler.read_from_memory(memory_handler.address.player2_hp)
  connection.client:writeU8(player2_hp)
end

function M:send_screenshot(connection)
  connection.client:writeU8(SendType.SCREENSHOT)
  local screenshot = PCSX.GPU.takeScreenShot()
  connection.client:writeI32(screenshot.height)
  connection.client:writeI32(screenshot.width)

  local bpp = 16
  if tonumber(screenshot.bpp) == 1 then
    bpp = 24
  end
  connection.client:writeU8(bpp)

  screenshot.data = tostring(screenshot.data)
  connection.client:writeU32(#screenshot.data)
  connection.client:write(screenshot.data)
end

return M
