import request from '@/utils/request'

export function queryAllLog() {
  return request({
    url: 'http://127.0.0.1:8080/log/queryAllLog',
    method: 'get'
  })
}
