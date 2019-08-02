import logging
import logging.handlers

#Logger 인스턴스 생성 및 로그 레벨 설정
logger = logging.getLogger("mylogger")
logger.setLevel(logging.DEBUG)

#formmater생성
#formatter=Logging.Formatter('[%(Levelname)s%(filename)s:%(lineno)s)]%(asctime)s>%(message)s')
formatter = logging.Formatter('[%(filename)s:%(lineno)s)]%(asctime)s>%(message)s')

#filehander 와 StreamHandler생성
fileMaxByte = 1024*1024*100
fileHandler = logging.handlers.RotatingFileHandler('../log/my.log', maxBytes=fileMaxByte, backupCount=10, encoding='utf-8')
streamHandler = logging.StreamHandler()

#handler에 formmater세팅
fileHandler.setFormatter(formatter)
streamHandler.setFormatter(formatter)

#Handler를 logging에 추가
logger.addHandler((fileHandler))
logger.addHandler(streamHandler)