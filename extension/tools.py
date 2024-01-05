import subprocess
import time
import os; rootdir = os.path.split(__file__)[0]
# rootdir = './extension'

def number_in_line(line):
    wordlist = line.split(' ')
    for _, item in enumerate(wordlist):
        try: number = float(item) 
        except ValueError: continue
        
    return number


def pc_error(infile1, infile2, resolution, normal=False, show=False):
    # print('Test distortion\t......')
    # print('resolution:\t', resolution)
    start_time = time.time()
    # headersF = ["mse1      (p2point)", "mse1,PSNR (p2point)", 
    #            "h.       1(p2point)", "h.,PSNR  1(p2point)",
    #            "mse2      (p2point)", "mse2,PSNR (p2point)", 
    #            "h.       2(p2point)", "h.,PSNR  2(p2point)" ,
    #            "mseF      (p2point)", "mseF,PSNR (p2point)", 
    #            "h.        (p2point)", "h.,PSNR   (p2point)" ]
    # headersF_p2plane = ["mse1      (p2plane)", "mse1,PSNR (p2plane)",
    #                   "mse2      (p2plane)", "mse2,PSNR (p2plane)",
    #                   "mseF      (p2plane)", "mseF,PSNR (p2plane)"]             
    headers = ["mseF      (p2point)", "mseF,PSNR (p2point)"]
    command = str(rootdir+'/pc_error_d' + 
                          ' -a '+infile1+ 
                          ' -b '+infile2+ 
                          ' --hausdorff=1 '+ 
                          ' --resolution='+str(resolution))
    if normal:
        headers +=["mseF      (p2plane)", "mseF,PSNR (p2plane)"]
        command = str(command + ' -n ' + infile1)
    results = {}   
    subp=subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    c=subp.stdout.readline()
    while c:
        line = c.decode(encoding='utf-8')# python3.
        if show: print(line)
        for _, key in enumerate(headers):
            if line.find(key) != -1:
                value = number_in_line(line)
                results[key] = value
        c=subp.stdout.readline() 
    # print('Test Distortion Done.', '\tTime:', round(time.time() - start_time, 3), 's')

    return results
