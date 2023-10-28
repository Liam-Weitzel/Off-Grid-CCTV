LDRpin = 0
IRCutFilterOn = 1
IRCutFilterOff = 2
IRLEDPin = 3
CamUpPin = 4
CamDownPin = 5 
CamRightPin = 6
CamLeftPin = 7
CamMiddlePin = 8 

GPIOpins = {1, 2, 3, 4, 5, 6, 7, 8}
GPIOstates = {true, true, true, true, true, true, true, true}

AutoNightVisionTmr = tmr.create()
nightVisionLDRThresh = 0
AutoNightVisionState = false
apName = "The Ultimate CCTV Camera"
apPwd = "Just_A_Default_Password"

function init()
  for k, v in pairs(GPIOpins) do --set GPIOpins to high
    gpio.mode(v,gpio.OUTPUT)
    gpio.write(v,gpio.HIGH)
  end

  togRelay(IRCutFilterOn) --Making sure the ir cam cut filter switch is in the right position
  tmr.delay(100000)
  togRelay(IRCutFilterOn)

  read() --Read and set saved states from the config file
end

function togRelay(number)
  if GPIOstates[number]==false then 
    gpio.write(GPIOpins[number],gpio.HIGH)
  else 
    gpio.write(GPIOpins[number],gpio.LOW) 
  end
  GPIOstates[number] = not GPIOstates[number];
end

function togNightVision()
    togRelay(IRLEDPin)
    if(GPIOstates[IRLEDPin]) then
        togRelay(IRCutFilterOn)
        tmr.delay(100000)
        togRelay(IRCutFilterOn)
    else
        togRelay(IRCutFilterOff)
        tmr.delay(100000)
        togRelay(IRCutFilterOff)
    end
end

function autoNightVision()
    if(not AutoNightVisionState) then
        AutoNightVisionTmr:alarm(1000, tmr.ALARM_AUTO, function()
           if(adc.read(LDRpin) <= tonumber(nightVisionLDRThresh) and GPIOstates[IRLEDPin] ~= false) then togNightVision() end
           if(adc.read(LDRpin) > tonumber(nightVisionLDRThresh) and GPIOstates[IRLEDPin] ~= true) then togNightVision() end
        end)
    end
    if(AutoNightVisionState) then
        AutoNightVisionTmr:unregister()
    end
    AutoNightVisionState = not AutoNightVisionState
end

function read()
    if file.open("config.txt", "r") then
        line = file.readline()
        while line ~= nil do
            print(line)
            varToSet = string.sub(line,0,string.find(line,":"))
            val = string.sub(line,string.find(line,":")+1)
            val = string.gsub(val, "\n", "")
            if(varToSet == "nightVisionLDRThresh:") then 
                nightVisionLDRThresh = tonumber(val)
            end
            if(varToSet == "AutoNightVisionState:") then 
                if(val ~= tostring(AutoNightVisionState)) then autoNightVision() end
            end
            if(varToSet == "apName:") then
                apName = val
            end
            if(varToSet == "apPwd:") then
                apPwd = val
            end
            line = file.readline()
        end
        file.close()
    end
end

function save(ldrVal, apNameTemp, apPwdTemp)
    nightVisionLDRThresh = ldrVal
    apName = apNameTemp
    apPwd = apPwdTemp
    if file.open("config.txt", "w+") then
        file.writeline("nightVisionLDRThresh:"..tostring(nightVisionLDRThresh))
        file.writeline("AutoNightVisionState:"..tostring(AutoNightVisionState))
        file.writeline("apName:"..apName)
        file.writeline("apPwd:"..apPwd)
        file.close()
    end
end

init()
wifi.setmode(wifi.STATIONAP) -- Start access point
wifi.ap.config({ssid=apName,pwd=apPwd})
print("Server IP Address:",wifi.ap.getip())
sv = net.createServer(net.TCP) -- Start webserver

sv:listen(80, function(conn)  
  conn:on("receive",function(conn,payload)
  print(payload)

    local function readControl()
      if payload ~= nil then
        fndLDR = {string.find(payload,"ldr=")}
        fndLDREnd = {string.find(payload,"&", fndLDR[2])}
        fndName = {string.find(payload,"apName=")}
        fndNameEnd = {string.find(payload,"&", fndName[2])}
        fndPwd = {string.find(payload,"apPwd=")}
        fndPwdEnd = {string.find(payload,"&", fndPwd[2])}
        fndControl = {string.find(payload,"control=")}
        fndControlEnd = {string.find(payload,"&", fndControl[2])}
        
        if(fndLDREnd[2] ~= nil) then ldrVal = string.sub(payload,fndLDR[2]+1,fndLDREnd[2]-1)
        else ldrVal = string.sub(payload,fndLDR[2]+1) end
        if(fndNameEnd[2] ~= nil) then apNameTemp = string.sub(payload,fndName[2]+1,fndNameEnd[2]-1)
        else apNameTemp = string.sub(payload,fndName[2]+1) end
        if(fndPwdEnd[2] ~= nil) then apPwdTemp = string.sub(payload,fndPwd[2]+1,fndPwdEnd[2]-1)
        else apPwdTemp = string.sub(payload,fndPwd[2]+1) end
        if(fndControlEnd[2] ~= nil) then control = string.sub(payload,fndControl[2]+1, fndControlEnd[2]-1)
        else control = string.sub(payload,fndControl[2]+1) end
        
        print(control)
        if control == "Toggle+night+vision" then togNightVision() return end
        if control == "Automatically+enable+night+vision" then autoNightVision() return end
        if control == "Save" then save(ldrVal, apNameTemp, apPwdTemp) return end
      end
    end

    LDRpercent = adc.read(LDRpin)/10

    --get control data from payload
    payloadExists = {string.find(payload,"control=")}
    if #payloadExists ~= 0 then readControl() end -- Is there data in payload? - Take action if so.

    conn:send('<!DOCTYPE HTML><html><body>\n')
    conn:send('<h1>Camera control panel</h1>\n')
    
    conn:send('<form action="" method="POST">\n')
    conn:send('<input type="submit" name="control" value="Toggle night vision"><br><br>\n')
    conn:send('<input type="submit" name="control" value="Toggle black and white filter"><br><br>\n')
    conn:send('<input type="submit" name="control" value="Automatically enable night vision"><br><br>\n')
    conn:send('<input type="submit" name="control" value="Automatically enable black and white filter when night vision is active"><br><br>\n')
    conn:send('Threshold for night vision: <input type="range" name="ldr" min="0" max="1000" value="'..nightVisionLDRThresh..'">\n')
    conn:send('<p> Current amount of light detected: '..LDRpercent..'%</p>\n')
    
    conn:send('<br><br>Set acces point name: <input name="apName" type="text" value="'..apName..'">\n')
    conn:send('<br><br>Set acces point password: <input name="apPwd" type="password" value="'..apPwd..'">\n')
    conn:send('*Will only take effect on next restart\n')

    conn:send('<br><br><br><br>Advanced: interface directly with camera settings')
    conn:send('<div style="padding: 20px; padding-left: 150px; width: 100px; height: 100px; display: grid; grid-template-columns: auto auto auto;">\n')
    conn:send('<div style="text-align:center"></div>\n')
    conn:send('<div style="text-align:center"><i style="border: solid black; border-width:0 3px 3px 0; display: inline-block; padding:3px;transform: rotate(-135deg);-webkit-transform: rotate(-135deg);"></i></div>\n')
    conn:send('<div style="text-align:center"></div>\n')
    conn:send('<div style="text-align:center"><i style="border: solid black; border-width:0 3px 3px 0; display: inline-block; padding:3px;transform: rotate(135deg);-webkit-transform: rotate(135deg);"></i></div>\n')
    conn:send('<div style="text-align:center"><i style="border: solid black; border-width:2px 2px 2px 2px; display: inline-block; padding:3px;"></i></div>\n')
    conn:send('<div style="text-align:center"><i style="border: solid black; border-width:0 3px 3px 0; display: inline-block; padding:3px;transform: rotate(-45deg);-webkit-transform: rotate(-45deg);"></i></div>\n')
    conn:send('<div style="text-align:center"></div>\n')
    conn:send('<div style="text-align:center"><i style="border: solid black; border-width:0 3px 3px 0; display: inline-block; padding:3px;transform: rotate(45deg);-webkit-transform: rotate(45deg);"></i></input></div>\n')
    conn:send('<div style="text-align:center"></div>\n')
    conn:send('</div>') 
    
    conn:send('<input type="submit" name="control" value="Save">\n')
    conn:send('</form></body></html>\n')
    
    conn:on("sent",function(conn) conn:close() end)
  end)
end)
