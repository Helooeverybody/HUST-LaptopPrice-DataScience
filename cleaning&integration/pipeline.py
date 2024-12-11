import pandas as pd
import re
import numpy as np

class Cleaner():
    def __init__(self):
        self.column_dict = {
            'link': self._nop,
            'name': self._name,
            'Display': self._display,
            'HDD/SSD': self._disk,
            'RAM': self._ram,
            'OS': self._os,
            'Body Material': self._material,
            'Dimensions': self._dimensions,
            'Weight': self._weight,
            'M.2 Slot': self._nvme,
            'USB Type-C': self._usbc,
            'USB Type-A': self._usba,
            'HDMI': self._hdmi,
            'Bluetooth': self._bluetooth,
            'Card Reader': self._cardreader,
            'Ethernet LAN': self._ethernet,
            'Security Lock slot': self._securitylock,
            'Fingerprint reader': self._fingerprint,
            'Backlit keyboard': self._backlit,
            'Cost': self._cost,
            'Portability Score': self._scores,
            'Play Score': self._scores,
            'Work Score': self._scores,
            'Display Score': self._scores,
            'Total Score': self._scores,
            'CPU': self._nop,
            'GPU': self._nop
        }

    def clean(self, df):
        cols = []
        df = df.copy().astype('str')
        for column, func in self.column_dict.items():
            if column in df.columns:
                cols.append(func(df[column]))
            else:
                print('column not found: ', column)
        df_out = pd.concat(cols, axis=1)
        return df_out

    def _nop(self, column):
        return column
    
    def _display(self, column):
        display = column.str.split(',', expand=True).to_numpy()
        for row in display:
            if row[3] is None:
                if row[2] is not None and not((row[2].find('Hz') != -1 or row[2].isnumeric())):
                    row[3] = row[2]
                    row[2] = None
        display = pd.DataFrame(display)
        # dealing with column 0, minor error in some collumns, convert them all to float
        display0 = display[0].str.extract(r'(\d+\.?\d+)', expand=True).astype(float)
        display0.rename(columns={0: 'Display Size'}, inplace=True)
        display0.fillna(15.6, inplace=True)
        # dealing with column 1, since resolution quality approximates its length*height, convert them all to float
        display1 = display[1].str.extract(r'(\d+)\s?x\s?(\d+)', expand=True).astype(float)
        display1.rename(columns={0: 'Resolution: width', 1: 'Resolution: height'}, inplace=True)
        display1['Resolution: height'].fillna(1080, inplace=True)
        display1['Resolution: width'].fillna(1920, inplace=True)
        # dealing with column 2, extract refresh rate and convert them all to float
        display2 = display[2].str.extract(r'(\d+)', expand=True).astype(float)
        display2.rename(columns={0: 'Refresh Rate'}, inplace=True)
        # fill NaN with 60, since 90 is the min in the known data, I assmue 60 is the missing since it's usually ignore in information sites
        display2.fillna(60, inplace=True)
        # dealing with column 3, leave alone
        display3 = display[3]
        display3.fillna('IPS', inplace=True)
        display3.rename('Panel Type', inplace=True)
        # gather them back
        display = pd.concat([display0, display1, display2, display3], axis=1)
        return display

    def _disk(self, column):
        disk = column.str.split('+' ,expand=True)[[0, 1]]
        part_1 = disk[0].astype(str)
        part_2 = disk[1].astype(str)
        result = np.zeros((len(part_1), 5))

        def extract(lst):
            if lst.find('tb') != -1:
                res = re.findall(r'(\d+)\s?tb', lst)
                num = float(res[0]) * 1000
            elif lst.find('gb') != -1:
                res = re.findall(r'(\d+)\s?gb', lst)
                num = float(res[0])
            else:
                num = -1
            if lst.find('ssd') != -1:
                tp = 'SSD'
            elif lst.find('hdd') != -1:
                tp = 'HDD'
            elif lst.find('sshd') != -1:
                tp = 'SSHD'
            elif lst.find('optane') != -1:
                tp = 'Optane'
            else:
                tp = -1
            return num, tp

        for i in range(len(part_1)):
            n = 0
            t = []
            lst = part_1.iloc[i].lower()
            if lst != 'None':
                a, b = extract(lst)
                if a != -1:
                    n += a
                if b != -1:
                    t.append(b)
            lst = part_2.iloc[i].lower()
            if lst != 'None':
                a, b = extract(lst)
                if a != -1:
                    n += a
                if b != -1:
                    t.append(b)
            if n != 0:
                result[i][0] = n
            # fill na for disk capacity
            else:
                result[i][0] = 512
            for j in t:
                match j:
                    case 'SSD':
                        result[i][1] = 1
                    case 'HDD':
                        result[i][2] = 1
                    case 'SSHD':
                        result[i][3] = 1
                    case 'Optane':
                        result[i][4] = 1
            if not t:
                # fill na for disk type
                result[i][1] = 1
                result[i][2] = 0
                result[i][3] = 0
                result[i][4] = 0
            
        result = pd.DataFrame(result, columns=['Disk Capacity', 'SSD', 'HDD', 'SSHD', 'Optane'])
        return result

    def _ram(self, column):
        ram = column.astype('str').str.split(',', expand=True)

        capacity = ram[0].str.extract(r'(\d+)', expand=True).astype(float)
        capacity.rename(columns={0: 'RAM Capacity'}, inplace=True)

        ram_type = ram[1]
        ram_type.rename('RAM Type', inplace=True)

        ram = pd.concat([capacity, ram_type], axis=1)
        return ram

    def _os(self, column):
        os = column.str.lower()
        return os
    
    def _dimensions(self, column):
        dim = column.astype('str')
        res = dim.str.extract(r'.*\s?x\s?.*\s?x\s?.*\s?x\s?(\d+\.?\d+)', expand=True).astype(float)
        res = res.apply(lambda x: x*2.54)
        dim = res.rename(columns={0: 'Dimension: Depth'})
        return dim

    def _material(self, column):
        material = column.str.lower().astype('str')
        res = {}

        for row in material:
            row = row.lower()
            r = row.split(',')
            r = [x.strip() for x in r]
            for w in r:
                if w not in res:
                    res[w] = len(res)

        aaaaa = np.zeros((len(material), len(res)))
        for i in range(len(material)):
            r = material.iloc[i]
            r = r.lower()
            r = r.split(',')
            r = [x.strip() for x in r]
            for w in r:
                aaaaa[i][res[w]] = 1
        material = pd.DataFrame(aaaaa, columns=['Body material: ' + x for x in res.keys()])
        material = material.drop('Body material: nan', axis=1)
        return material

    def _weight(self, column):
        weight = column.str.extract(r'(\d+\.?\d+)', expand=True).astype(float)
        weight.rename(columns={0: 'Weight'}, inplace=True)
        return weight

    def _nvme(self, column):
        nvme = column.astype('str')
        res = nvme.str.extract(r'(\d+)\s?x', expand=True).astype(float)
        nvme = res.rename(columns={0: 'Num of M.2 Slot'})
        nvme.fillna(0, inplace=True)
        return nvme

    def _usbc(self, column):
        usb_c = column.astype('float')
        usb_c.fillna(0, inplace=True)
        return usb_c

    def _usba(self, column):
        usb_a = column.astype('float')
        usb_a.fillna(0, inplace=True)
        return usb_a

    def _hdmi(self, column):
        hdmi = column.astype('float')
        hdmi.fillna(0, inplace=True)
        return hdmi

    def _bluetooth(self, column):
        bluetooth = column.astype('float')
        bluetooth.fillna(0, inplace=True)
        return bluetooth

    def _cardreader(self, column):
        card_reader = column.astype('str')
        res = card_reader.apply(lambda x: 0 if x == 'nan' else (0 if x.lower() == '0' else 1))
        card_reader = res.rename('Card Reader')
        return card_reader
                
    def _ethernet(self, column):
        ethernet = column.astype('str')
        for i in range(len(ethernet)):
            if ethernet.iloc[i] == 'None' or ethernet.iloc[i] == 'nan':
                ethernet.iloc[i] = np.nan
            else:
                res = re.findall(r'(\d+)', ethernet.iloc[i])
                ethernet.iloc[i] = res[-1]

        ethernet = ethernet.astype(float)
        return ethernet

    def _securitylock(self, column):
        security = column.astype('str')

        for i in range(len(security)):
            if security.iloc[i] == 'None' or security.iloc[i] == 'nan' or security.iloc[i] == '0':
                security.iloc[i] = 0
            else:
                security.iloc[i] = 1

        security = security.astype(float)
        return security
                
    def _fingerprint(self, column):
        fingerprint = column.astype('float')
        fingerprint.fillna(0, inplace=True)
        return fingerprint

    def _backlit(self, column):
        backlit = column.astype('str')

        for i in range(len(backlit)):
            if backlit.iloc[i] == 'None' or backlit.iloc[i] == 'nan':
                backlit.iloc[i] = 0
            elif backlit.iloc[i] == '0':
                backlit.iloc[i] = 0
            else:
                backlit.iloc[i] = 1
                
        backlit = backlit.astype(float)
        return backlit

    def _scores(self, column):
        name = column.name
        column = column.str.extract(r'(\d+)', expand=True).astype(float)
        column.rename(columns={0: name}, inplace=True)
        return column
                
    def _cost(self, column):
        cost = column.str.extract(r'(\d+)', expand=True).astype(float)
        cost.rename(columns={0: 'Cost'}, inplace=True)
        return cost

    def _name(self, column):
        name = column.str.lower()
        return name

class Integrator():
    def __init__(self, cpu, gpu):
        self.cpu = cpu
        self.gpu = gpu
        self.cpu = self.cpu.set_index('CPU: Name')
        self.gpu = self.gpu.set_index('GPU: Name')

    def integrate(self, lap):
        lap['CPU'] = lap['CPU'].str.lower()
        lap['GPU'] = lap['GPU'].str.lower()
        out_cpu = self._cpu(lap)
        out_gpu = self._gpu(lap)
        pd_out = pd.concat([lap, out_cpu, out_gpu], axis=1)
        pd_out.drop('CPU', inplace=True)
        pd_out.drop('GPU', inplace=True)
        return pd_out
    
    def _cpu(self, lap):
        # Process cpu and gpu name to defined format
        trim = {'amd':'', 'intel':'', 'apple':'', 'qualcomm':'', 'nvidia':''}
        l_cpu = lap['CPU'].astype(str)
        for i in range(len(l_cpu)):
            for key in trim:
                l_cpu[i] = l_cpu[i].replace(key, trim[key]).strip().lower()
        # CPU and GPU: concat from the 2 datasets
        result_cpu = {}
        cpu_not_found = 0
        for i in range(len(lap)):
            cur_cpu = l_cpu[i]
            try:
                res = self.cpu.loc[cur_cpu].to_numpy()
                if len(res.shape) > 1:
                    print('CPU:', cur_cpu, 'found multiple:', res)
                    res = res[0]
                res = np.insert(res, 0, cur_cpu)
                result_cpu[i] = res
            except:
                cpu_not_found += 1
                result_cpu[i] = np.array([np.nan]*(len(self.cpu.columns)+1))
                print('CPU not found:', cur_cpu)
        print('CPU not found:', cpu_not_found)
        cpu_cols = self.cpu.columns.insert(0, 'CPU: Name')
        result_cpu = pd.DataFrame.from_dict(result_cpu, orient='index', columns=cpu_cols)
        return result_cpu

    def _gpu(self, lap):
        # Process cpu and gpu name to defined format
        trim = {'amd':'', 'intel':'', 'apple':'', 'qualcomm':'', 'nvidia':''}
        l_gpu = lap['GPU'].astype(str)
        for i in range(len(l_gpu)):
            for key in trim:
                l_gpu[i] = l_gpu[i].replace(key, trim[key]).strip().lower()
        # CPU and GPU: concat from the 2 datasets
        result_gpu = {}
        gpu_not_found = 0
        for i in range(len(lap)):
            cur_gpu = l_gpu[i]
            try:
                res = self.gpu.loc[cur_gpu].to_numpy()
                if len(res.shape) > 1:
                    print('GPU:', cur_gpu, 'found multiple:', res)
                    res = res[0]
                res = np.insert(res, 0, cur_gpu)
                result_gpu[i] = res
            except:
                gpu_not_found += 1
                result_gpu[i] = np.array([np.nan]*(len(self.gpu.columns)+1))
                print('GPU not found:', cur_gpu)
        print('GPU not found:', gpu_not_found)
        gpu_cols = self.gpu.columns.insert(0, 'GPU: Name')
        result_gpu = pd.DataFrame.from_dict(result_gpu, orient='index', columns=gpu_cols)
        return result_gpu
        
        

if __name__ == '__main__':
    data_path = '../data/'
    df = pd.read_csv(data_path + 'Laptop-partially-cleaned.csv')
    cpu = pd.read_csv(data_path + 'cpu_merged.csv')
    gpu = pd.read_csv(data_path + 'gpu_merged.csv')
    cleaner = Cleaner()
    df_out = cleaner.clean(df)
    integrator = Integrator(cpu, gpu)
    df_out = integrator.integrate(df_out)
    df_out.to_csv(data_path + 'pipeline.csv')