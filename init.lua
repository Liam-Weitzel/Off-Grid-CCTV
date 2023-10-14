
LEDpins = {0,4}
LEDstate = true
RELAYpins = {5, 6, 7, 8}
RELAYstates = {true, true, true, true}
UNUSEDpins = {1,2,3}
LDRval = 0
LDRpin = 0
NightVisionState = true

function init()
  for k, v in pairs(LEDpins) do --set LEDpins to HIGH
    gpio.mode(v,gpio.OUTPUT)
    gpio.write(v,gpio.HIGH)
  end
  
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
  
  blinkOFF()
  tmr.create():alarm(1000, tmr.ALARM_AUTO, function() -- read LDR val
    LDRval = adc.read(LDRpin)
  end)
end

function togRelay(number)
  if RELAYstates[number]==false then 
    gpio.write(RELAYpins[number],gpio.HIGH)
  else 
    gpio.write(RELAYpins[number],gpio.LOW) 
  end
  RELAYstates[number] = not RELAYstates[number];
end

function togLED()
  if LEDstate==false then 
    for k, v in pairs(LEDpins) do --set LEDpins to HIGH
      gpio.write(v,gpio.HIGH)
    end
  else 
    for k, v in pairs(LEDpins) do --set LEDpins to LOW
      gpio.write(v,gpio.LOW)
    end
  end
  LEDstate = not LEDstate;
end

function blkinkON()
  if mytimer~=nil then -- Timer already on.
    return 
  end 
  mytimer = tmr.create()
  mytimer:alarm(200, tmr.ALARM_AUTO, function() togLED() end)
end

function blinkOFF()
  if mytimer==nil then -- Timer already off. 
    return 
  end 
  mytimer:unregister() mytimer=nil
end

function togNightVision()
  togRelay(2)
  if(NightVisionState) then
    togRelay(3)
    tmr.delay(10000)
    togRelay(3)
  else 
    togRelay(4)
    tmr.delay(100000)
    togRelay(4)
  end
  NightVisionState = not NightVisionState
end

wifi.setmode(wifi.STATIONAP) -- Start access point
wifi.ap.config({ssid="test",pwd="password"})
print("Server IP Address:",wifi.ap.getip())
sv = net.createServer(net.TCP) -- Start webserver
init()

sv:listen(80, function(conn)  
  conn:on("receive",function(conn,payload)
    print(payload)
    function controlLED()
      if payload ~= nil then
        control = string.sub(payload,fnd[2]+1) -- Data is at end already.
        if control == "LED" then togLED(); blinkOFF() return end
        if control == "Blink" then blkinkON() return end
        if control == "Blinkoff" then blinkOFF() return end
        if control == "NightVision" then togNightVision() return end
      end
    end
    --get control data from payload
    fnd = {string.find(payload,"control=")}
    if #fnd ~= 0 then controlLED() end -- Is there data in payload? - Take action if so.

    conn:send('<!DOCTYPE HTML>\n')
    conn:send('<html>\n')
    conn:send('<head><meta http-equiv="content-type" content="text/html; charset=UTF-8">\n')
    -- Scale the viewport to fit the device.
    conn:send('<meta name="viewport" content="width=device-width, initial-scale=1">')
    -- Title
    conn:send('<title>ESP8266 Wifi LED Control</title>\n')
    -- CSS style definition for submit buttons
    conn:send('<style>\n')
    conn:send('input[type="submit"] {\n')
    conn:send('color:#050; width:70px; padding:10px;\n')
    conn:send('font: bold 84% "trebuchet ms",helvetica,sans-serif;\n')
    conn:send('background-color:lightgreen;\n')
    conn:send('border:1px solid; border-radius: 12px;\n')
    conn:send('transition-duration: 0.4s;\n')
    conn:send('}\n')
    conn:send('input[type="submit"]:hover {\n')
    conn:send('background-color:lightblue;\n')
    conn:send('color: white;\n')
    conn:send('}')
    conn:send('</style></head>\n')
    -- HTML body Page content.
    conn:send('<body>')
    conn:send('<h1>Control of nodeMCU<br>(ESP8266-E12) Built in LED.</h1>\n')
    conn:send('<p>The built in LED for NodeMCU V3 is on D4</p>\n')
    -- HTML Form (POST type) and buttons.
    conn:send('<form action="" method="POST">\n')
    conn:send('<input type="submit" name="control" value="LED" > Toggle Built in LED <br><br>\n')
    conn:send('<input type="submit" name="control" value="Blink"> Blink LED <br><br>\n')
    conn:send('<input type="submit" name="control" value="Blinkoff"> Stop LED Blink<br><br>\n')
    conn:send('<input type="submit" name="control" value="NightVision" > Toggle night vision </form>\n')
    conn:send('<p> LDR out: '..LDRval..'</p>')
    conn:send('</body></html>\n')

    conn:on("sent",function(conn) conn:close() end)
  end)
end)
