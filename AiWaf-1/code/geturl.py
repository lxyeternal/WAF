# _*_ coding:utf-8 _*_

'''
author:   guowenbo
time:     2019/1/4
us:       sichuanunivesity
'''

from scapy.all import *
from urllib.parse import unquote
from train_url import *
try:
    # This import works from the project directory
    import scapy_http.http as http
except ImportError:
    # If you installed this package via pip, you just need to execute this
    from scapy.layers import http

test = []
def Sniffer():

    rule = 'tcp port 80'

    # prn = lambda x: x[IP].src
    # lambda x: x.summary()
    # <script>alert(1)<script>

    http_sniffer = sniff(filter=rule,iface= 'en0',prn = pack_callback,count=1)
    # wrpcap("en0sniff.pcap", http_sniffer)

def pack_callback(packet):

    pack_for_url = []
    if 'TCP' in packet:
        Ether_dst = 'Mac_Dst：' + packet.dst
        Ether_src = 'Mac_Src：' + packet.src
        IP_src = 'IP_Src：' + packet.payload.src
        IP_dst = 'IP_Dst：' + packet.payload.dst

        if packet.haslayer(http.HTTPRequest):

            http_header = packet[http.HTTPRequest].fields
            http_method = 'Method：' + http_header['Method'].decode()
            http_agent = 'User-Agent：' + http_header['User-Agent'].decode()
            http_host = 'Host：' + http_header['Host'].decode()
            http_path = 'Path：' + http_header['Path'].decode()
            deal_good_url = 'URL：' + unquote(http_header['Host'].decode() + http_header['Path'].decode())

            if deal_good_url:

                print(IP_src)
                print(IP_dst)
                print(Ether_src)
                print(Ether_dst)
                print(deal_good_url)
                print(http_method)
                print(http_agent)
                print(http_host)
                print(http_path)
                pack_for_url.append(IP_src)
                pack_for_url.append(IP_dst)
                pack_for_url.append(Ether_src)
                pack_for_url.append(Ether_dst)
                pack_for_url.append(deal_good_url)
                pack_for_url.append(http_method)
                pack_for_url.append(http_agent)
                pack_for_url.append(http_host)
                pack_for_url.append(http_path)
                test.append(pack_for_url)
                print(pack_for_url)

        elif packet.haslayer(http.HTTPResponse):
            pass

    return pack_for_url
