#测试pybook13_1_SaveToSQLiteMod的效果
from PythonFu import pybook13_1_SaveToSQLiteMod, pybook13_1_DataForUSDA

WhatFileName='food.db'
WhereOpen='D:\\boyLen\\py\\Pylearn1\\dataLen\\NUTR_DEF2.txt'
WhatExecute='update {tablename} set A=?,B=?,C=?,D=?,E=? where id =?'
tablename='food4'
pybook13_1_SaveToSQLiteMod.GetTxtAndUpdateSQLite_6NUM(WhatFileName, WhereOpen, WhatExecute, tablename)
pybook13_1_DataForUSDA.CreateAndInsertData_64W()