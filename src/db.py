import pymysql

db = pymysql.connect(host='175.210.42.85', port=3306, user='root', password='root123', db='text', charset='utf8')

cursor = db.cursor(pymysql.cursors.DictCursor)
sql = "select version()"
cursor.execute(sql)
result = cursor.fetchall()
result