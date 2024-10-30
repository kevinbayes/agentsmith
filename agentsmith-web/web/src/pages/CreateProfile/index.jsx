import React from 'react';
import {useNavigate,} from 'react-router-dom';
import { withAuth0 } from '@auth0/auth0-react';
import BaseComponent from "../../components/BaseComponent/index.jsx";
import {Bloc} from "./bloc.js";
import {
  Alert,
  Avatar,
  Box, Button, Container, CssBaseline, Grid, LinearProgress, Link, TextField,
} from "@mui/material";
import {withAppContext,} from "../context.jsx";
import {H1Header} from "../../components/Typography/index.jsx";
import {Copyright, LockOutlined} from "@mui/icons-material";
import FormControlLabel from "@mui/material/FormControlLabel";
import Checkbox from "@mui/material/Checkbox";

class CreateProfile extends BaseComponent {

  constructor(props) {
    super(props);
    this.setBloc(new Bloc({ user: props.auth0.user, }, props.globalContext.bloc));
    this.state = { initialised: false };
  }

  componentDidMount() {
    super.componentDidMount();
    this.bloc.initialise();
  }

  render() {

    const { initialised, processing, form, error } = this.state;

    if(!initialised) {
      return <div>Loading...</div>;
    }

    return <Container component="main" maxWidth="xs">
      { processing && <LinearProgress /> }
      <Box
          sx={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
          }}
      >
        <Avatar sx={{ m: 1, bgcolor: 'secondary.main' }}>
          <LockOutlined />
        </Avatar>
        <H1Header>Complete Registration</H1Header>
        <Box component="form"  onSubmit={this.bloc.submit} sx={{ mt: 3 }}>
          <Grid container spacing={2}>
            <Grid item xs={12}>
              { error && <Alert severity="error">{ error }</Alert> }
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                  autoComplete="given-name"
                  name="given_name"
                  required
                  fullWidth
                  id="given_name"
                  label="Given name"
                  value={form.given_name}
                  onChange={this.bloc.text_changed}
                  autoFocus
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                  required
                  fullWidth
                  id="family_name"
                  label="Family name"
                  name="family_name"
                  value={form.family_name}
                  autoComplete="family-name"
                  onChange={this.bloc.text_changed}
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                  required
                  fullWidth
                  id="email"
                  label="Email Address"
                  name="email"
                  value={form.email}
                  autoComplete="email"
                  onChange={this.bloc.text_changed}
              />
            </Grid>
            <Grid item xs={12}>
              <FormControlLabel
                  control={<Checkbox required={true} name={"terms"} onChange={this.bloc.checkbox_changed} value={form.terms} color="primary" />}
                  label="I accept the terms and conditions"
              />
            </Grid>
            <Grid item xs={12}>
              <FormControlLabel
                  control={<Checkbox required={true} name={"email_consent"}  onChange={this.bloc.checkbox_changed} value={form.email_consent} color="primary" />}
                  label="I want to receive product updates and communication via email."
              />
            </Grid>
          </Grid>
          <Button
              type="submit"
              fullWidth
              variant="contained"
              sx={{ mt: 3, mb: 2 }}
              disabled={processing}
          >
            Complete
          </Button>
        </Box>
      </Box>
    </Container>;
  }
}

export default withAuth0(withAppContext(CreateProfile));