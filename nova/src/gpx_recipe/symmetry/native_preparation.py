"""Lossless native bookkeeping normalization on a private project copy."""
import copy


def parameter_flags(parameters,index):
    result=copy.deepcopy(parameters)
    for name,value in result.items():
        if hasattr(value,'tolist'):value=value.tolist();result[name]=value
        if isinstance(value,list) and len(value)>index and type(value[index]) in {int,float,bool} and value[index] in {0,1}:
            value[index]=bool(value[index])
    return result


def normalize_flags(project):
    """Older GPXs encode refinement flags as 0/1; preserve numerical values."""
    rows=[]
    def flag(values,index,path):
        if isinstance(values,list) and len(values)>index and type(values[index]) in {int,float} and values[index] in {0,1}:
            rows.append(dict(parameter=path,flag_integer=values[index]));values[index]=bool(values[index])
    for h in project.histograms():
        for family,index in [('Instrument Parameters',2),('Sample Parameters',1)]:
            values=h.data[family][0] if family=='Instrument Parameters' else h.data[family]
            new=parameter_flags(values,index)
            for name,value in values.items():
                if isinstance(value,list) and len(value)>index and type(value[index]) is int and value[index] in {0,1}:
                    rows.append(dict(histogram=h.name,family=family,parameter=name,flag_integer=value[index]))
            values.clear();values.update(new)
        background=h.data.get('Background',[])
        if background:flag(background[0],1,h.name+'/background')
        if len(background)>1:
            for term in background[1].get('debyeTerms',[]):
                for i in [1,3,5]:flag(term,i,h.name+'/Debye')
            for peak in background[1].get('peaksList',[]):
                for i in [4,5,6,7]:flag(peak,i,h.name+'/background peak')
    for phase in project.phases():
        flag(phase.data['General']['Cell'],0,phase.name+'/cell')
        for name,hap in phase.data['Histograms'].items():
            for key,index in [('Scale',1),('Pref.Ori.',2),('Extinction',1)]:
                if key in hap:flag(hap[key],index,phase.name+'/'+name+'/'+key)
            for family in ['Size','Mustrain']:
                for index in [2,5]:
                    if family in hap and len(hap[family])>index:
                        for i in range(len(hap[family][index])):flag(hap[family][index],i,phase.name+'/'+name+'/'+family)
            if 'HStrain' in hap:
                for i in range(len(hap['HStrain'][1])):flag(hap['HStrain'][1],i,phase.name+'/'+name+'/HStrain')
            for term in hap.get('Babinet',{}).values():flag(term,1,phase.name+'/'+name+'/Babinet')
    return rows
