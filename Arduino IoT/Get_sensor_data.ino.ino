#include <HTTPClient.h>

#include <WiFi.h>
#include <WiFiClient.h>

// Replace with your WiFi credentials
const char* ssid = "67TND";
const char* password = "0909094005";

// ThingSpeak MQTT broker details
const char* mqttServer = "mqtt.thingspeak.com";
const char* mqttUsername = "DQcqIgkRAiUfDTUYCh4yLA4";
const char* mqttPassword = "5ntzi3Pvgo/aqb9YecZi3uCf";

// ThingSpeak channel details
const char* channelID = "2344323";
const char* apiKey = "IMT17H98LGVWW9QH";

// Select the Analog pins to read the sensors
#define FLAME_SENSOR_PIN 35
#define GAS_SENSOR_PIN 34

// Store the Analog values read from the sensors
int flameSensorValue;
int gasSensorValue;

// Store the values (%) converted from the corresponding Analog values
int flamePercent;
int gasPercent;

// Flag to control sensor readings
bool readSensors = false;

void setup() {
  Serial.begin(115200);

  // Connect to Wi-Fi
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    Serial.println("Connecting to WiFi...");
  }

  Serial.println("Connected to WiFi");
  Serial.println(WiFi.localIP());
}

void loop() {
  // Read the command from the Serial Monitor.
  String command = Serial.readStringUntil('\n');

  // Check if the command is "start" to enable sensor readings.
  if (command == "start") {
    readSensors = true;
    Serial.println("Sensor readings started.");
  }
    if (command == "stop") {
    readSensors = false;
    Serial.println("Sensor readings stopped.");
  }

  // Read sensors if enabled
  if (readSensors) {
    // Read analog value from the flame sensor
    flameSensorValue = analogRead(FLAME_SENSOR_PIN);
    // Convert flame sensor value to a percentage scale
    flamePercent = map(flameSensorValue, 0, 623, 0, 100);
    // Transmit the measured value of the flame sensor to the computer.
    Serial.print("Flame Detection in %: ");
    Serial.println(flamePercent);

    // Read analog value from the gas sensor
    gasSensorValue = analogRead(GAS_SENSOR_PIN);
    // Convert gas sensor value to a percentage scale
    gasPercent = map(gasSensorValue, 0, 623, 0, 100);
    // Transmit the measured value of the gas sensor to the computer.
    Serial.print("Gas Detector in %: ");
    Serial.println(gasPercent);

    // Publish sensor values to ThingSpeak channel
    publishToThingSpeak(flamePercent, gasPercent);

    // Delay before the next reading
    delay(3000);
  }
}

void publishToThingSpeak(int flameValue, int gasValue) {
  // Create a WiFiClient object to establish a connection to ThingSpeak
  WiFiClient client;

  // Construct the ThingSpeak API URL
  String url = "http://api.thingspeak.com/update?api_key=" + String(apiKey) +
               "&field1=" + String(flameSensorValue) + "&field2=" + String(gasSensorValue);

  Serial.print("Sending HTTP request to ThingSpeak: ");
  Serial.println(url);

  // Connect to ThingSpeak
  if (client.connect("api.thingspeak.com", 80)) {
    // Send HTTP GET request
    client.println("GET " + url + " HTTP/1.1");
    client.println("Host: api.thingspeak.com");
    client.println("Connection: close");
    client.println();

    // Wait for the server's response
    while (client.available()) {
      char c = client.read();
      Serial.write(c);
    }
    

    Serial.println("HTTP request sent successfully");
    delay(5000);
    client.stop();
  } else {
    Serial.println("Failed to connect to ThingSpeak");
  }

}