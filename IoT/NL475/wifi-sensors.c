/* WiFi Example
 * Copyright (c) 2016 ARM Limited
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "mbed.h"
#include "TCPSocket.h"
// Sensors drivers present in the BSP library
#include "stm32l475e_iot01_tsensor.h"
#include "stm32l475e_iot01_hsensor.h"
#include "stm32l475e_iot01_psensor.h"
//#include "stm32l475e_iot01_magneto.h"
//#include "stm32l475e_iot01_gyro.h"
//#include "stm32l475e_iot01_accelero.h"


#define WIFI_ESP8266    1
#define WIFI_IDW0XX1    2

#if TARGET_UBLOX_EVK_ODIN_W2
#include "OdinWiFiInterface.h"
OdinWiFiInterface wifi;

#elif TARGET_REALTEK_RTL8195AM
#include "RTWInterface.h"
RTWInterface wifi;

#elif TARGET_DISCO_L475VG_IOT01A
#include "ISM43362Interface.h"
ISM43362Interface wifi(MBED_CONF_APP_WIFI_SPI_MOSI, MBED_CONF_APP_WIFI_SPI_MISO, MBED_CONF_APP_WIFI_SPI_SCLK, MBED_CONF_APP_WIFI_SPI_NSS, MBED_CONF_APP_WIFI_RESET, MBED_CONF_APP_WIFI_DATAREADY, MBED_CONF_APP_WIFI_WAKEUP, false);

#else // External WiFi modules

#if MBED_CONF_APP_WIFI_SHIELD == WIFI_ESP8266
#include "ESP8266Interface.h"
ESP8266Interface wifi(MBED_CONF_APP_WIFI_TX, MBED_CONF_APP_WIFI_RX);
#elif MBED_CONF_APP_WIFI_SHIELD == WIFI_IDW0XX1
#include "SpwfSAInterface.h"
SpwfSAInterface wifi(MBED_CONF_APP_WIFI_TX, MBED_CONF_APP_WIFI_RX);
#endif // MBED_CONF_APP_WIFI_SHIELD == WIFI_IDW0XX1

#endif

const char *sec2str(nsapi_security_t sec)
{
    switch (sec) {
        case NSAPI_SECURITY_NONE:
            return "None";
        case NSAPI_SECURITY_WEP:
            return "WEP";
        case NSAPI_SECURITY_WPA:
            return "WPA";
        case NSAPI_SECURITY_WPA2:
            return "WPA2";
        case NSAPI_SECURITY_WPA_WPA2:
            return "WPA/WPA2";
        case NSAPI_SECURITY_UNKNOWN:
        default:
            return "Unknown";
    }
}

int scan_demo(WiFiInterface *wifi)
{
    WiFiAccessPoint *ap;

    printf("Scan:\n");

    int count = wifi->scan(NULL,0);
    printf("%d networks available.\n", count);

    /* Limit number of network arbitrary to 15 */
    count = count < 15 ? count : 15;

    ap = new WiFiAccessPoint[count];
    count = wifi->scan(ap, count);
    for (int i = 0; i < count; i++)
    {
        printf("Network: %s secured: %s BSSID: %hhX:%hhX:%hhX:%hhx:%hhx:%hhx RSSI: %hhd Ch: %hhd\n", ap[i].get_ssid(),
               sec2str(ap[i].get_security()), ap[i].get_bssid()[0], ap[i].get_bssid()[1], ap[i].get_bssid()[2],
               ap[i].get_bssid()[3], ap[i].get_bssid()[4], ap[i].get_bssid()[5], ap[i].get_rssi(), ap[i].get_channel());
    }

    delete[] ap;
    return count;
}

void http_demo(NetworkInterface *net)
{
    TCPSocket socket;
    nsapi_error_t response;

    printf("\nSending HTTP request to host...\n");

    
    printf("Connected with 34.246.10.0\n");
    unsigned idx = 0;
    while(1)
    {
    // Open a socket on the network interface, and create a TCP connection to www.arm.com
    //response = socket.connect("www.arm.com", 80);
        socket.open(net);
        response = socket.connect("34.246.10.0", 9000);
        if(0 != response) {
            printf("Error connecting: %d\n", response);
            socket.close();
            return;
        }        // Send a simple http request
        //char sbuffer[] = "GET / HTTP/1.1\r\nHost: www.arm.com\r\n\r\n";
        //wait(5);
        idx++;
        float temp = BSP_TSENSOR_ReadTemp();
        char s_temp[80];
        //sprintf(s_temp, "%.2fdegC", temp);
        sprintf(s_temp, "%.2f", temp);
        //puts(s_temp);
        
        float humi = BSP_HSENSOR_ReadHumidity();
        char s_humi[80];
        //sprintf(s_humi, "%.2f%%", humi);
        sprintf(s_humi, "%.2f", humi);
        //puts(s_humi);

        float pressure = BSP_PSENSOR_ReadPressure();
        char s_pre[80];
        //sprintf(s_pre, "%.2fmBar", pressure);
        sprintf(s_pre, "%.2f", pressure);
        //puts(s_pre);

        char s_idx[16];
        sprintf(s_idx, "%d", idx);
        //puts(s_idx);

        char qStr[256]="GET /?idx=";
        strcat(qStr, s_idx);  strcat(qStr,"&");
        strcat(qStr,"temp="); strcat(qStr,s_temp); strcat(qStr,"&");
        strcat(qStr,"humi="); strcat(qStr,s_humi); strcat(qStr,"&");
        strcat(qStr,"pressure="); strcat(qStr,s_pre); strcat(qStr," ");
        strcat(qStr, "HTTP/1.1\r\nHost: 34.246.10.0\r\n\r\n");
        //puts(qStr);

        //char sbuffer[] = "GET /test?data=1&temp=21.1&humi=60% HTTP/1.1\r\nHost: 192.168.1.193\r\n\r\n";
        nsapi_size_t size = strlen(qStr);
        response = 0;
        while(size)
        {
            response = socket.send(qStr+response, size);
            if (response < 0) {
                printf("Error sending data: %d\n", response);
                //socket.close();
                break;
            } else {
                size -= response;
                // Check if entire message was sent or not
                printf("\nsent %d [%.*s]\n", response, strstr(qStr, "\r\n")-qStr, qStr);
            }
        }

        // Recieve a simple http response and print out the response line
        /*
        char rbuffer[128];
        response = socket.recv(rbuffer, sizeof rbuffer);
        if (response <= 0) {
            printf("Error receiving data: %d\n", response);
        } else {
            printf("recv %d [%.*s]\n", response, strstr(rbuffer, "\r\n")-rbuffer, rbuffer);
        }
        */
        socket.close();
        wait(5);
    }
    // Close the socket to return its memory and bring down the network interface
    //socket.close();
}

int main()
{
    int count = 0;

    printf("WiFi example\n\n");

    count = scan_demo(&wifi);
    if (count == 0) {
        printf("No WIFI APNs found - can't continue further.\n");
        return -1;
    }

    printf("\nConnecting to %s...\n", MBED_CONF_APP_WIFI_SSID);
    int ret = wifi.connect(MBED_CONF_APP_WIFI_SSID, MBED_CONF_APP_WIFI_PASSWORD, NSAPI_SECURITY_WPA_WPA2);
    if (ret != 0) {
        printf("\nConnection error\n");
        return -1;
    }

    printf("Success\n\n");
    printf("MAC: %s\n", wifi.get_mac_address());
    printf("IP: %s\n", wifi.get_ip_address());
    printf("Netmask: %s\n", wifi.get_netmask());
    printf("Gateway: %s\n", wifi.get_gateway());
    printf("RSSI: %d\n\n", wifi.get_rssi());

    BSP_TSENSOR_Init();
    BSP_HSENSOR_Init();
    BSP_PSENSOR_Init();

    http_demo(&wifi);

    wifi.disconnect();

    printf("\nDone\n");
}

