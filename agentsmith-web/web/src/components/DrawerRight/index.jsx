import React from 'react';

import BaseComponent from "../BaseComponent/index.jsx";
import {Bloc} from "./bloc.js";
import {withAuth0} from "@auth0/auth0-react";
import {
  Box,
} from "@mui/material";
import {withAppContext} from "../../pages/context.jsx";



class DrawerRight extends BaseComponent {

  constructor(props) {
    super(props);
    this.setBloc(new Bloc({
      leftDrawer: 350,
    }, props.globalContext));

    this.state = {};
  }

  componentDidMount() {
    super.componentDidMount();
  }

  render() {

    const { initialised } = this.state;

    return (
        <Box>
          Left
        </Box>
    );
  }
}

export default withAuth0(withAppContext(DrawerRight));