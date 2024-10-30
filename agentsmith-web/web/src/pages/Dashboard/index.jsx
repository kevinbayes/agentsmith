import React from 'react';
import {withAuth0} from '@auth0/auth0-react';
import BaseComponent from "../../components/BaseComponent/index.jsx";
import {Bloc} from "./bloc.js";
import {Box, Card, Grid2, Paper, styled} from "@mui/material";

class Dashboard extends BaseComponent {

  constructor(props) {
    super(props);
    this.setBloc(new Bloc({auth0: props.auth0}));
  }

  __onTextChange = (event) => {
    this.bloc.setQuery(event.target.value);
  }

  __onKeyPress = (event) => {
    if (event.key === 'Enter') {
      event.preventDefault();
      this.__processQuery();
    }
  }

  __processQuery = () => {
    this.bloc.runQuery();
  }

  render() {
    return <Box sx={{ paddingRight: "56px" }}>
      <Box sx={{ padding: "16px" }}>
        <Grid2 spacing={6} container>
          Dashboard
        </Grid2>
      </Box>
    </Box>;
  }
}

export default withAuth0(Dashboard);