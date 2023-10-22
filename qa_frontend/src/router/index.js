import Vue from "vue";
import VueRouter from "vue-router";
// import Home from '../views/Home.vue'
import Index from "../views/Index.vue";

Vue.use(VueRouter);

const routes = [
  {
    path: "/",
    name: "Index",
    component: Index,
    meta: {
      title: "首页",
      requireAuth: true,
      index: 1,
    },
  },
  {
    path: "/chatbot",
    name: "Chatbot",
    // route level code-splitting
    // this generates a separate chunk (about.[hash].js) for this route
    // which is lazy-loaded when the route is visited.
    component: () =>
      import(/* webpackChunkName: "about" */ "../views/ChatBox.vue"),
    meta: {
      title: "聊天",
      requireAuth: true,
      index: 2,
    },
  },
];

const router = new VueRouter({
  mode: "history",
  base: process.env.BASE_URL,
  routes,
});

export default router;
