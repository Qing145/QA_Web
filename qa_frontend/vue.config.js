module.exports = {
    devServer: {
      port: 8080,
      open: true,
      hot: true, //自动保存
      proxy: {
        '/api': {
          target: 'http://127.0.0.1:9999',
          changeOrigin: true,
          pathRewrite: {
            '^/api': ''
          }
        }
      }
    }
  }