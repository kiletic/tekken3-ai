local M = {}

M.address = {
  player1_hp = 0x800a961e,
  player2_hp = 0x800aaeaa,
  match_status = 0x8009547c,
  game_paused = 0x8009548c
}

function M.read_from_memory(address)
  local mem = PCSX.getMemPtr()
  address = bit.band(address, 0x1fffff)
  local pointer = mem + address
  pointer = ffi.cast('uint32_t*', pointer)
  local value = pointer[0]
  return value, pointer
end

return M
