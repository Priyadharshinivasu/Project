import { Component } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import { UserComponent } from './User/user.component';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [RouterOutlet,UserComponent],
  templateUrl: './app.component.html',
  styleUrl: './app.component.css'
})
export class AppComponent {
  title = 'ShoppingMallApp';
}
