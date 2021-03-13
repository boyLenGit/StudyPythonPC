import segyio
from segyio import TraceField


read_path = 'E:/Research/data/F3_entire.segy'
with segyio.open(read_path) as F3_entire:
    print('--------------------------------- #Others ---------------------------------')
    print('###### attributes:', F3_entire.attributes('traces'))
    # print('######flush:', F3_entire.flush())
    print('###### mmap:', F3_entire.mmap())
    print('###### dtype:', F3_entire.dtype)
    print('###### fast:', F3_entire.fast)
    print('###### format:', F3_entire.format)
    print('###### tracecount:', F3_entire.tracecount)
    print('--------------------------------- #lines ---------------------------------')
    print('###### iline:', F3_entire.iline)
    print('###### ilines--len:', len(F3_entire.ilines))
    print('###### xline:', F3_entire.xline)
    print('###### xlines--len:', len(F3_entire.xlines))
    print('--------------------------------- trace ---------------------------------')
    print('###### trace:', F3_entire.trace)
    print('###### len(F3_entire.trace):', len(F3_entire.trace))
    print('###### trace.raw:', F3_entire.trace.raw[:])
    print('###### trace.length:', F3_entire.trace.length)
    # print('######trace.count:', F3_entire.trace.count())  # ()中要加值才可以
    print('###### trace.ref:', F3_entire.trace.ref)
    print('###### trace.readonly:', F3_entire.trace.readonly)
    print('--------------------------------- header ---------------------------------')
    print('###### header:', F3_entire.header)
    print('###### header_len:', len(F3_entire.header))
    print('###### header[TraceField]:', F3_entire.header[10][TraceField.TRACE_SEQUENCE_LINE])
    # print('###### header.dic1:', F3_entire.header[0]['TRACE_SEQUENCE_LINE'])
    for i1 in range(5):
        print('###### header{0}:'.format(i1), F3_entire.header[i1])
    print('###### header[0][TraceField.offset]:', F3_entire.header[0][TraceField.offset])
    # print('###### header.iline:', F3_entire.header.iline.header.segy)  # 返回了SegyFile、inlines、crosslines、traces、samples等。全是数字
    # print('###### header.xline.lines:', F3_entire.header.xline.lines)  # 返回了一大串數字
    # print('###### header.xline:', F3_entire.header.xline.header.segy)
    # print('###### text:', F3_entire.text, type(F3_entire.text))  # √
    '''for i1 in range(len(F3_entire.text)):
        print('###### For{0}:'.format(i1), F3_entire.text[i1])'''
    print('###### ext_headers:', F3_entire.ext_headers)



