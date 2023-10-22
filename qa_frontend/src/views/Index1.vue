<template>
    <div id="index-root">
      <div id="chat-content" ref="scroll">
        <div v-for="(itemc, indexc) in recordContent" :key="indexc">
          <!-- 对方 -->
          <div class="word" v-if="itemc.id == 2">
            <img :src="itemc.headUrl" />
            <div class="info">
              <p class="time">
                {{ itemc.nickName }}
              </p>
              <div class="info-content">{{ itemc.contactText }}</div>
              <div class="message_time">
                {{ itemc.time }}
              </div>
            </div>
          </div>
  
          <!-- 我的 -->
          <div class="word-my" v-else>
            <div class="info">
              <p class="nickname">
                {{ itemc.nickName }}
              </p>
              <div class="info-content">{{ itemc.contactText }}</div>
              <div class="Sender_time">
                {{ itemc.time }}
              </div>
            </div>
            <img :src="itemc.headUrl" />
          </div>
        </div>
      </div>
      <div id="chat-input">
        <div class="chat-input-box">
          <el-input
            type="textarea"
            :rows="3"
            placeholder="Please input the question"
            v-model="textarea"
            resize="none"
            @keydown.enter.native="sendMessage"
          >
          </el-input>
        </div>
        <div class="chat-input-button">
          <el-button type="primary" @click="sendMessage">Send Message</el-button>
        </div>
      </div>
    </div>
  </template>
  
  <script>
  export default {
    name: "index",
    data() {
      return {
        activeIndex: "1",
        recordContent: [],
        textarea: "",
        currentTime: "",
      };
    },
    components: {},
    methods: {
      handleSelect(key, keyPath) {
        console.log(key, keyPath);
      },
      sendMessage(e) {
        // 将this存储在that，在this.axios会发生this对象指向问题
        let that = this;
        e.preventDefault();
  
        // 计算当前时间并按格式输出
        this.timeCal();
  
        // 构建聊天记录对象并插入记录数组中
        let obj = {
          id: 1,
          nickName: "User",
          contactText: this.textarea,
          headUrl: "/user.jpeg",
          time: this.currentTime,
        };
        this.recordContent.push(obj);
        let question = this.textarea;
        this.textarea = "";
  
        // 将问题发给后端并返回答案
        this.$axios
          .post("http://127.0.0.1:5000/getAnswer", {
            question: question,
          })
          .then((res) => {
            let answer = res.data;
  
            that.timeCal();
  
            let obj = {
              id: 2,
              nickName: "Doctor",
              contactText: answer,
              headUrl: "/doctor.jpeg",
              time: that.currentTime,
            };
  
            that.recordContent.push(obj);
          });
      },
      timeCal() {
        // 获取时间并按格式拼接
        let date = new Date();
        let year = date.getFullYear();
        let month = date.getMonth() + 1;
        let day = date.getDate();
        let hour = date.getHours();
        let minute = date.getMinutes();
  
        // 小于10的小时和分钟都在前面加上0
        if (hour < 10) {
          hour = "0" + hour;
        }
        if (minute < 10) {
          minute = "0" + minute;
        }
  
        let date_format =
          hour + ":" + minute + " " + month + "/" + day + "/" + year;
  
        this.currentTime = date_format;
      },
    },
    watch: {
      recordContent() {
        // 等下一次dom更新之后再执行的函数
        this.$nextTick(() => {
          let container = this.$refs.scroll;
          container.scrollTo({
            top: container.scrollHeight,
            behavior: "smooth",
          });
        });
      },
    },
    beforeMount() {
      this.timeCal();
  
      let obj = {
        id: 2,
        nickName: "Doctor",
        contactText: "您好，我是医药智能助理小可爱，希望可以帮到您。祝您身体健康！",
        headUrl: "/doctor.jpeg",
        time: this.currentTime,
      };
  
      this.recordContent.push(obj);
    },
  };
  </script>
  
  <style lang="less" scoped>
  #index-root {
    height: 100%;
    overflow: hidden;
    display: flex;
    flex-direction: column;
    #chat-content {
      box-sizing: border-box;
      width: 100%;
      padding: 30px;
      background: #f3f3f3;
      height: 100%;
      overflow: auto;
      .word {
        display: flex;
        margin-bottom: 10px;
  
        img {
          width: 60px;
          height: 60px;
          border-radius: 50%;
        }
        .info {
          width: 47%;
          margin-left: 10px;
          text-align: left;
          .message_time {
            font-size: 12px;
            color: rgba(51, 51, 51, 0.8);
            margin: 0;
            height: 20px;
            line-height: 20px;
            margin-top: -5px;
            margin-top: 5px;
          }
          .info-content {
            word-break: break-all;
            // max-width: 45%;
            display: inline-block;
            padding: 10px;
            font-size: 14px;
            background: #fff;
            position: relative;
            margin-top: 8px;
            border-radius: 4px;
          }
          //小三角形
          .info-content::before {
            position: absolute;
            left: -8px;
            top: 8px;
            content: "";
            border-right: 10px solid #fff;
            border-top: 8px solid transparent;
            border-bottom: 8px solid transparent;
          }
        }
      }
  
      .word-my {
        display: flex;
        justify-content: flex-end;
        margin-bottom: 10px;
        img {
          width: 60px;
          height: 60px;
          border-radius: 50%;
        }
        .info {
          width: 90%;
          // margin-left: 10px;
          text-align: right;
          // position: relative;
          display: flex;
          align-items: flex-end;
          flex-wrap: wrap;
          flex-direction: column;
          .nickname {
            margin-right: 10px;
          }
          .info-content {
            word-break: break-all;
            max-width: 45%;
            padding: 10px;
            font-size: 14px;
            margin-right: 10px;
            position: relative;
            margin-top: 8px;
            background: #97e24c;
            text-align: left;
            border-radius: 4px;
          }
          .Sender_time {
            padding-right: 12px;
            padding-top: 5px;
            font-size: 12px;
            color: rgba(51, 51, 51, 0.8);
            margin: 0;
            height: 20px;
          }
          //小三角形
          .info-content::after {
            position: absolute;
            right: -8px;
            top: 8px;
            content: "";
            border-left: 10px solid #97e24c;
            border-top: 8px solid transparent;
            border-bottom: 8px solid transparent;
          }
        }
      }
    }
    #chat-content::-webkit-scrollbar {
      display: none;
    }
    #chat-input {
      height: 139px;
      text-align: right;
      flex-grow: 0;
      flex-shrink: 0;
      border-top: 1px solid #c7c7c7;
      .chat-input-box {
        box-sizing: border-box;
        padding: 10px;
      }
      .chat-input-button {
        display: inline-block;
        box-sizing: border-box;
        padding: 0 10px;
      }
    }
  }
  </style>