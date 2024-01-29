import request from '@/utils/request'

export function getPeerNodesCount() {
  return request({
    url: 'http://127.0.0.1:80/api/dashboard/count/peer',
    method: 'get'
  })
}

export function getOrganizations() {
  return request({
    url: 'http://127.0.0.1:80/api/dashboard/count/organization',
    method: 'get'
  })
}

export function getOrdererNodesCount() {
  return request({
    url: 'http://127.0.0.1:80/api/dashboard/count/orderer',
    method: 'get'
  })
}

export function getCACount() {
  return request({
    url: 'http://127.0.0.1:80/api/dashboard/count/ca',
    method: 'get'
  })
}

export function getChannel() {
  return request({
    url: 'http://127.0.0.1:80/api/dashboard/count/channel',
    method: 'get'
  })
}

export function getBlockHeight() {
  return request({
    url: 'http://127.0.0.1:80/api/dashboard/count/blockHeight',
    method: 'get'
  })
}

export function getTransactionCount() {
  return request({
    url: 'http://127.0.0.1:80/api/dashboard/count/transaction',
    method: 'get'
  })
}

export function getChaincodeCount() {
  return request({
    url: 'http://127.0.0.1:80/api/dashboard/count/chaincode',
    method: 'get'
  })
}
