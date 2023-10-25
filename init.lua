
--Memory remedies:
--get rid of float reads on pins
--see if i can get rid of more libraries in my binary file

RELAYpins = {5, 6, 7, 8}
RELAYstates = {true, true, true, true}
UNUSEDpins = {0,1,2,3,4}
LDRpin = 0
AutoNightVisionTmr = tmr.create()
nightVisionLDRThresh = 0
NightVisionState = false
AutoNightVisionState = false

function init()
  for k, v in pairs(RELAYpins) do --set RELAYpins to high
    gpio.mode(v,gpio.OUTPUT)
    gpio.write(v,gpio.HIGH)
  end

  for k, v in pairs(UNUSEDpins) do --set UNUSEDpins to low
    gpio.mode(v,gpio.OUTPUT)
    gpio.write(v,gpio.LOW)
  end --Getting them out of floating point saves power

  togRelay(4) --Making sure the ir cam cut filter switch is in the right position
  tmr.delay(100000)
  togRelay(4)

  read() --Read and set saved states from the config file
end

function togRelay(number)
  if RELAYstates[number]==false then 
    gpio.write(RELAYpins[number],gpio.HIGH)
  else 
    gpio.write(RELAYpins[number],gpio.LOW) 
  end
  RELAYstates[number] = not RELAYstates[number];
end

function togNightVision()
  togRelay(2)
  if(NightVisionState) then
    togRelay(4)
    tmr.delay(100000)
    togRelay(4)
  else 
    togRelay(3)
    tmr.delay(100000)
    togRelay(3)
  end
  NightVisionState = not NightVisionState
end

function autoNightVision()
    if(not AutoNightVisionState) then
        AutoNightVisionTmr:alarm(1000, tmr.ALARM_AUTO, function()
           if(adc.read(LDRpin) <= tonumber(nightVisionLDRThresh) and NightVisionState == false) then togNightVision() end
           if(adc.read(LDRpin) > tonumber(nightVisionLDRThresh) and NightVisionState == true) then togNightVision() end
        end)
    end
    if(AutoNightVisionState) then
        AutoNightVisionTmr:unregister()
    end
    AutoNightVisionState = not AutoNightVisionState
end

function togCamera()
    togRelay(1) 
    if(NightVisionState) then 
        togNightVision() 
    end
    if(AutoNightVisionState) then
        autoNightVision()
    end
end

function read()
    if file.open("config.txt", "r") then
        line = file.readline()
        while line ~= nil do
            print(line)
            varToSet = string.sub(line,0,string.find(line,":"))
            val = string.sub(line,string.find(line,":")+1)
            if(varToSet == "nightVisionLDRThresh:") then 
                nightVisionLDRThresh = tonumber(val)
            end
            if(varToSet == "NightVisionState:") then 
                if(tostring(val) ~= tostring(NightVisionState).."\n") then togNightVision() end
            end
            if(varToSet == "AutoNightVisionState:") then 
                if(tostring(val) ~= tostring(AutoNightVisionState).."\n") then autoNightVision() end
            end
            if(varToSet == "CameraState:") then 
                if(tostring(val) ~= tostring(RELAYstates[1]).."\n") then togCamera() end
            end
            line = file.readline()
        end
        file.close()
    end
end

function save(ldrVal)
    nightVisionLDRThresh = ldrVal
    if file.open("config.txt", "w+") then
        file.writeline("nightVisionLDRThresh:"..tostring(nightVisionLDRThresh))
        file.writeline("NightVisionState:"..tostring(NightVisionState))
        file.writeline("AutoNightVisionState:"..tostring(AutoNightVisionState))
        file.writeline("CameraState:"..tostring(RELAYstates[1]))
        file.close()
    end
end

wifi.setmode(wifi.STATIONAP) -- Start access point
wifi.ap.config({ssid="test",pwd="password"})
print("Server IP Address:",wifi.ap.getip())
sv = net.createServer(net.TCP) -- Start webserver
init()

sv:listen(80, function(conn)  
  conn:on("receive",function(conn,payload)
  print(payload)

    local function readControl()
      if payload ~= nil then
        fndLDR = {string.find(payload,"ldr=")}
        ldrVal = string.sub(payload,fndLDR[2]+1)
        control = string.sub(payload,fndCONTROL[2]+1, fndLDR[2]-5) -- Data is at end already.
        print(control)
        if control == "Toggle+night+vision" then togNightVision() return end
        if control == "Toggle+camera" then togCamera() return end
        if control == "Automatically+enable+night+vision" then autoNightVision() return end
        if control == "Save" then save(ldrVal) return end
      end
    end

    local function isDisabled() 
        if(AutoNightVisionState) then return "button" end
        if(not AutoNightVisionState) then return "submit" end
    end

    LDRpercent = adc.read(LDRpin)/10

    --get control data from payload
    fndCONTROL = {string.find(payload,"control=")}
    if #fndCONTROL ~= 0 then readControl() end -- Is there data in payload? - Take action if so.

        conn:send('<!DOCTYPE HTML>\n')
        conn:send('<html>\n')
        conn:send('<head><meta http-equiv="content-type" content="text/html; charset=UTF-8">\n')
        -- Scale the viewport to fit the device.
        conn:send('<meta name="viewport" content="width=device-width, initial-scale=1">')
        -- Title
        conn:send('<title>Camera control panel</title>\n')
        -- CSS style definition for submit buttons
        conn:send('<style>\n')
        conn:send('input[type="submit"] {\n')
        conn:send('padding:10px;\n')
        conn:send('font: bold 84% "trebuchet ms",helvetica,sans-serif;\n')
        conn:send('border:1px solid; border-radius: 12px;\n')
        conn:send('transition-duration: 0.4s;\n')
        conn:send('}\n')
        conn:send('input[type="submit"].false {\n background-color:lightred; color: red;}\n')
        conn:send('input[type="submit"].true {\n background-color:lightgreen; color: green;}\n') 
        conn:send('input[type="submit"]:hover {\n')
        conn:send('background-color:lightblue;\n')
        conn:send('color: white;\n')
        conn:send('}')
        conn:send('input[type="button"] {\n')
        conn:send('color: grey;padding:10px;background-color: lightgrey;\n')
        conn:send('font: bold 84% "trebuchet ms",helvetica,sans-serif;\n')
        conn:send('border:1px solid; border-radius: 12px;\n')
        conn:send('transition-duration: 0.4s;\n')
        conn:send('}\n')
        conn:send('</style></head>\n')
        -- HTML body Page content.
        conn:send('<body>')
        conn:send('<h1>Camera control panel</h1>\n')
        -- HTML Form (POST type) and buttons.
        conn:send('<form action="" method="POST">\n')
        conn:send('<input class="'..tostring(RELAYstates[1])..'" type="submit" name="control" value="Toggle camera"><br><br>\n')
        conn:send('<input class="'..tostring(NightVisionState)..'" type="'..isDisabled()..'" name="control" value="Toggle night vision"><br><br>\n')
        conn:send('<input class="'..tostring(AutoNightVisionState)..'" type="submit" name="control" value="Automatically enable night vision"><br><br>\n')
        conn:send('<input type="submit" name="control" value="Save">\n')
        conn:send('Threshold for night vision: <input type="range" name="ldr" min="0" max="1000" value="'..nightVisionLDRThresh..'"></form>\n')
        conn:send('<p> Current amount of light detected: '..LDRpercent..'%</p>\n')
        conn:send('</body></html>\n')

    conn:on("sent",function(conn) conn:close() end)
  end)
end)
