import request from '@/utils/request'

export function queryAllChannel() {
  return request({
    url: 'http://127.0.0.1:8080/overview/queryAllChannel',
    method: 'get'
  })
}

export function queryAllOrg() {
  return request({
    url: 'http://127.0.0.1:8080/overview/queryAllOrg',
    method: 'get'
  })
}

export function queryAllChaincode() {
  return request({
    url: 'http://127.0.0.1:8080/overview/queryAllChaincode',
    method: 'get'
  })
}
