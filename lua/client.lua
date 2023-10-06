local network_parser = require("scripts/protocol")

local ADDR = "127.0.0.1"
local PORT = 50007

local M = {}

M.connected = false
M.client = nil

function M:toggle_status(name)
  local changed
  changed, self.connected = imgui.Checkbox(name, self.connected)
  if changed then
    if self.connected then
      self.client = Support.File.uvFifo(ADDR, PORT)
    else
      print("Client closed")
      self.client:close()
    end
  end

  if self.connected then
    if not self.client:isConnecting() and self.client:failed() then
      print("Client failed to connect")
      self.connected = false
    else
      if self.client:size() > 0 then
        network_parser:parse(self, self.client:readU8())
      end
    end
  end
end

return M
